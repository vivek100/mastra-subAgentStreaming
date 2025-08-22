import type { ToolCallOptions } from '@ai-sdk/provider-utils-v5';
import type { WritableStream } from 'stream/web';

import {
  OpenAIReasoningSchemaCompatLayer,
  OpenAISchemaCompatLayer,
  GoogleSchemaCompatLayer,
  AnthropicSchemaCompatLayer,
  DeepSeekSchemaCompatLayer,
  MetaSchemaCompatLayer,
  applyCompatLayer,
  convertZodSchemaToAISDKSchema,
} from '@mastra/schema-compat';
import type { ToolExecutionOptions } from 'ai';
import { z } from 'zod';
import { MastraBase } from '../../base';
import { ErrorCategory, MastraError, ErrorDomain } from '../../error';
import { RuntimeContext } from '../../runtime-context';
import { isVercelTool } from '../../tools/toolchecks';
import type { ToolOptions } from '../../utils';
import { ToolStream } from '../stream';
import type { CoreTool, ToolAction, VercelTool, VercelToolV5, SubAgentStreamingConfig } from '../types';
import { validateToolInput } from '../validation';
import { ChunkFrom } from '../../stream/types';

export type ToolToConvert = VercelTool | ToolAction<any, any, any> | VercelToolV5;
export type LogType = 'tool' | 'toolset' | 'client-tool';

interface LogOptions {
  agentName?: string;
  toolName: string;
  type?: 'tool' | 'toolset' | 'client-tool';
}

interface LogMessageOptions {
  start: string;
  error: string;
}

export class CoreToolBuilder extends MastraBase {
  private originalTool: ToolToConvert;
  private options: ToolOptions;
  private logType?: LogType;

  constructor(input: { originalTool: ToolToConvert; options: ToolOptions; logType?: LogType }) {
    super({ name: 'CoreToolBuilder' });
    this.originalTool = input.originalTool;
    this.options = input.options;
    this.logType = input.logType;
  }

  // Helper to get parameters based on tool type
  private getParameters = () => {
    if (isVercelTool(this.originalTool)) {
      return this.originalTool.parameters ?? z.object({});
    }

    return this.originalTool.inputSchema ?? z.object({});
  };

  private getOutputSchema = () => {
    if ('outputSchema' in this.originalTool) return this.originalTool.outputSchema;
    return null;
  };

  // For provider-defined tools, we need to include all required properties
  private buildProviderTool(tool: ToolToConvert): (CoreTool & { id: `${string}.${string}` }) | undefined {
    if (
      'type' in tool &&
      tool.type === 'provider-defined' &&
      'id' in tool &&
      typeof tool.id === 'string' &&
      tool.id.includes('.')
    ) {
      const parameters = this.getParameters();
      const outputSchema = this.getOutputSchema();
      return {
        type: 'provider-defined' as const,
        id: tool.id,
        args: ('args' in this.originalTool ? this.originalTool.args : {}) as Record<string, unknown>,
        description: tool.description,
        parameters: convertZodSchemaToAISDKSchema(parameters),
        ...(outputSchema ? { outputSchema: convertZodSchemaToAISDKSchema(outputSchema) } : {}),
        execute: this.originalTool.execute
          ? this.createExecute(
              this.originalTool,
              { ...this.options, description: this.originalTool.description },
              this.logType,
            )
          : undefined,
      };
    }

    return undefined;
  }

  private createLogMessageOptions({ agentName, toolName, type }: LogOptions): LogMessageOptions {
    // If no agent name, use default format
    if (!agentName) {
      return {
        start: `Executing tool ${toolName}`,
        error: `Failed tool execution`,
      };
    }

    const prefix = `[Agent:${agentName}]`;
    const toolType = type === 'toolset' ? 'toolset' : 'tool';

    return {
      start: `${prefix} - Executing ${toolType} ${toolName}`,
      error: `${prefix} - Failed ${toolType} execution`,
    };
  }

  private createExecute(tool: ToolToConvert, options: ToolOptions, logType?: 'tool' | 'toolset' | 'client-tool') {
    // dont't add memory or mastra to logging
    const { logger, mastra: _mastra, memory: _memory, runtimeContext, ...rest } = options;

    const { start, error } = this.createLogMessageOptions({
      agentName: options.agentName,
      toolName: options.name,
      type: logType,
    });

    const execFunction = async (args: unknown, execOptions: ToolExecutionOptions | ToolCallOptions) => {
      if (isVercelTool(tool)) {
        return tool?.execute?.(args, execOptions as ToolExecutionOptions) ?? undefined;
      }

      // Resolve optional sub-agent streaming config (ToolAction only)
      let subAgentStreaming: SubAgentStreamingConfig | undefined;
      const isToolAction = (obj: any): obj is ToolAction => obj && typeof obj === 'object' && 'execute' in obj;
      if (isToolAction(tool) && 'subAgentStreaming' in tool && tool.subAgentStreaming) {
        try {
          subAgentStreaming =
            typeof tool.subAgentStreaming === 'function'
              ? (tool.subAgentStreaming as any)({ context: args })
              : tool.subAgentStreaming;
        } catch {
          // ignore invalid config to preserve backwards-compat
          subAgentStreaming = undefined;
        }
      }

      // If not enabled, behave exactly as before
      const originalWritable = (options.writableStream || (execOptions as any).writableStream) as
        | WritableStream<import('../../stream/types').ChunkType>
        | undefined;

      const toolWriter = new ToolStream(
        {
          prefix: 'tool',
          callId: (execOptions as any).toolCallId,
          name: options.name,
          runId: options.runId!,
        },
        originalWritable,
      );

      // Early return when no sub-agent streaming config or disabled
      if (!subAgentStreaming || subAgentStreaming.enabled !== true) {
        return (
          tool?.execute?.(
            {
              context: args,
              threadId: options.threadId,
              resourceId: options.resourceId,
              mastra: options.mastra,
              memory: options.memory,
              runId: options.runId,
              runtimeContext: options.runtimeContext ?? new RuntimeContext(),
              writer: toolWriter,
            },
            execOptions as ToolExecutionOptions & ToolCallOptions,
          ) ?? undefined
        );
      }

      // Normalized config with defaults and guards
      const normalizedConfig: SubAgentStreamingConfig = {
        enabled: true,
        depth: subAgentStreaming.depth === 2 ? 2 : 1,
        streamToolCalls: subAgentStreaming.streamToolCalls !== false,
        streamText: subAgentStreaming.streamText === true,
        toolCallPrefix: subAgentStreaming.toolCallPrefix,
        contextMetadata: subAgentStreaming.contextMetadata,
      };

      // Writer for sub-agent forwarding - writes directly to parent stream using our new sub-* events
      const subEventWriter = originalWritable
        ? {
            write: async (chunk: any) => {
              const writer = originalWritable.getWriter();
              try {
                await writer.write(chunk);
              } finally {
                writer.releaseLock();
              }
            },
          }
        : undefined;

      // Construct a mastra proxy that intercepts getAgent().streamVNext and getAgent().generate to forward chunks
      const parentAgentName = options.agentName;
      const parentToolName = options.name;
      const parentRunId = options.runId!;

      const makeContext = (overrides?: Partial<import('../../stream/types').SubAgentStreamContext>) => ({
        depth: 1,
        parentRunId,
        parentAgentName,
        parentToolName,
        toolCallPrefix: normalizedConfig.toolCallPrefix,
        metadata: normalizedConfig.contextMetadata,
        ...overrides,
      });

      const forwardSubAgentStream = async (agent: any, input: any, streamOptions?: any) => {
        // Only depth 1 supported now; if configured 2, allow sub-agent to create its own nested events (handled elsewhere if added)
        const subStream = await agent.streamVNext(input, streamOptions);

        // Iterate over MastraModelOutput.fullStream to capture runId
        const fullStream: ReadableStream<any> = (subStream as any).fullStream;
        let subRunId = parentRunId;

        if (!fullStream) {
          // No full stream available; emit start with parent run id
          await subEventWriter?.write?.({
            type: 'sub-agent-start',
            runId: subRunId,
            from: ChunkFrom.AGENT,
            payload: { agentName: agent.name, prompt: input },
            context: makeContext({ depth: 1 }),
          });

          // Fallback: try textStream only
          try {
            if (normalizedConfig.streamText && (subStream as any).textStream) {
              for await (const t of (subStream as any).textStream as AsyncIterable<string>) {
                await subEventWriter?.write?.({
                  type: 'sub-text',
                  runId: subRunId,
                  from: ChunkFrom.AGENT,
                  payload: { text: t },
                  context: makeContext({ depth: 1 }),
                });
              }
            }
          } finally {
            await subEventWriter?.write?.({
              type: 'sub-agent-end',
              runId: subRunId,
              from: ChunkFrom.AGENT,
              payload: { finalResult: await (subStream as any).text?.catch?.(() => undefined) },
              context: makeContext({ depth: 1 }),
            });
          }

          return subStream;
        }

        const reader = fullStream.getReader();
        // Prime the reader to get runId
        let firstRead = await reader.read();
        if (!firstRead.done && firstRead.value && firstRead.value.runId) {
          subRunId = firstRead.value.runId;
        }

        // Emit start with sub-agent run id
        await subEventWriter?.write?.({
          type: 'sub-agent-start',
          runId: subRunId,
          from: ChunkFrom.AGENT,
          payload: { agentName: agent.name, prompt: input },
          context: makeContext({ depth: 1 }),
        });

        try {
          // Process the first chunk if present
          if (!firstRead.done) {
            const chunk = firstRead.value;
            if (chunk && chunk.type) {
              switch (chunk.type) {
                case 'tool-call':
                  if (normalizedConfig.streamToolCalls) {
                    await subEventWriter?.write?.({
                      type: 'sub-tool-call',
                      runId: subRunId,
                      from: ChunkFrom.AGENT,
                      payload: {
                        toolCallId: chunk.payload?.toolCallId,
                        toolName: normalizedConfig.toolCallPrefix
                          ? `${normalizedConfig.toolCallPrefix}.${chunk.payload?.toolName}`
                          : chunk.payload?.toolName,
                        args: chunk.payload?.args,
                      },
                      context: makeContext({ depth: 1 }),
                    });
                  }
                  break;
                case 'tool-result':
                  if (normalizedConfig.streamToolCalls) {
                    await subEventWriter?.write?.({
                      type: 'sub-tool-result',
                      runId: subRunId,
                      from: ChunkFrom.AGENT,
                      payload: {
                        toolCallId: chunk.payload?.toolCallId,
                        toolName: normalizedConfig.toolCallPrefix
                          ? `${normalizedConfig.toolCallPrefix}.${chunk.payload?.toolName}`
                          : chunk.payload?.toolName,
                        result: chunk.payload?.result,
                        isError: chunk.payload?.isError,
                      },
                      context: makeContext({ depth: 1 }),
                    });
                  }
                  break;
                case 'text-delta':
                  if (normalizedConfig.streamText && chunk.payload?.text) {
                    await subEventWriter?.write?.({
                      type: 'sub-text',
                      runId: subRunId,
                      from: ChunkFrom.AGENT,
                      payload: { text: chunk.payload.text },
                      context: makeContext({ depth: 1 }),
                    });
                  }
                  break;
                default:
                  break;
              }
            }
          }

          // Continue reading remaining chunks
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = value;
            if (!chunk || !chunk.type) continue;
            switch (chunk.type) {
              case 'tool-call':
                if (normalizedConfig.streamToolCalls) {
                  await subEventWriter?.write?.({
                    type: 'sub-tool-call',
                    runId: subRunId,
                    from: ChunkFrom.AGENT,
                    payload: {
                      toolCallId: chunk.payload?.toolCallId,
                      toolName: normalizedConfig.toolCallPrefix
                        ? `${normalizedConfig.toolCallPrefix}.${chunk.payload?.toolName}`
                        : chunk.payload?.toolName,
                      args: chunk.payload?.args,
                    },
                    context: makeContext({ depth: 1 }),
                  });
                }
                break;
              case 'tool-result':
                if (normalizedConfig.streamToolCalls) {
                  await subEventWriter?.write?.({
                    type: 'sub-tool-result',
                    runId: subRunId,
                    from: ChunkFrom.AGENT,
                    payload: {
                      toolCallId: chunk.payload?.toolCallId,
                      toolName: normalizedConfig.toolCallPrefix
                        ? `${normalizedConfig.toolCallPrefix}.${chunk.payload?.toolName}`
                        : chunk.payload?.toolName,
                      result: chunk.payload?.result,
                      isError: chunk.payload?.isError,
                    },
                    context: makeContext({ depth: 1 }),
                  });
                }
                break;
              case 'text-delta':
                if (normalizedConfig.streamText && chunk.payload?.text) {
                  await subEventWriter?.write?.({
                    type: 'sub-text',
                    runId: subRunId,
                    from: ChunkFrom.AGENT,
                    payload: { text: chunk.payload.text },
                    context: makeContext({ depth: 1 }),
                  });
                }
                break;
              default:
                break;
            }
          }
        } finally {
          await subEventWriter?.write?.({
            type: 'sub-agent-end',
            runId: subRunId,
            from: ChunkFrom.AGENT,
            payload: { finalResult: await (subStream as any).text?.catch?.(() => undefined) },
            context: makeContext({ depth: 1 }),
          });
        }

        return subStream;
      };

      // Mastra proxy that overrides getAgent to inject forwarding
      const injectedMastra = options.mastra
        ? new Proxy(options.mastra as any, {
            get(target, prop, receiver) {
              const value = Reflect.get(target, prop, receiver);
              if (prop === 'getAgent' && typeof value === 'function') {
                return new Proxy(value, {
                  apply(fn, thisArg, argArray) {
                    const agent = Reflect.apply(fn, thisArg, argArray);
                    if (agent && typeof agent.streamVNext === 'function') {
                      return new Proxy(agent, {
                        get(aTarget, aProp, aReceiver) {
                          const aValue = Reflect.get(aTarget, aProp, aReceiver);
                          if (aProp === 'streamVNext' && typeof aValue === 'function') {
                            return async function (...sArgs: any[]) {
                              return await forwardSubAgentStream(aTarget, sArgs[0], sArgs[1]);
                            };
                          }
                          return aValue;
                        },
                      });
                    }
                    return agent;
                  },
                });
              }
              return value;
            },
          })
        : options.mastra;

      return (
        tool?.execute?.(
          {
            context: args,
            threadId: options.threadId,
            resourceId: options.resourceId,
            mastra: injectedMastra,
            memory: options.memory,
            runId: options.runId,
            runtimeContext: options.runtimeContext ?? new RuntimeContext(),
            writer: toolWriter,
          },
          execOptions as ToolExecutionOptions & ToolCallOptions,
        ) ?? undefined
      );
    };

    return async (args: unknown, execOptions?: ToolExecutionOptions | ToolCallOptions) => {
      let logger = options.logger || this.logger;
      try {
        logger.debug(start, { ...rest, args });

        // Validate input parameters if schema exists
        const parameters = this.getParameters();
        const { data, error } = validateToolInput(parameters, args, options.name);
        if (error) {
          logger.warn(`Tool input validation failed for '${options.name}'`, {
            toolName: options.name,
            errors: error.validationErrors,
            args,
          });
          return error;
        }
        // Use validated/transformed data
        args = data;

        // there is a small delay in stream output so we add an immediate to ensure the stream is ready
        return await new Promise((resolve, reject) => {
          setImmediate(async () => {
            try {
              const result = await execFunction(args, execOptions!);
              resolve(result);
            } catch (err) {
              reject(err);
            }
          });
        });
      } catch (err) {
        const mastraError = new MastraError(
          {
            id: 'TOOL_EXECUTION_FAILED',
            domain: ErrorDomain.TOOL,
            category: ErrorCategory.USER,
            details: {
              errorMessage: String(error),
              argsJson: JSON.stringify(args),
              model: rest.model?.modelId ?? '',
            },
          },
          err,
        );
        logger.trackException(mastraError);
        logger.error(error, { ...rest, error: mastraError, args });
        return mastraError;
      }
    };
  }

  buildV5() {
    const builtTool = this.build();

    if (!builtTool.parameters) {
      throw new Error('Tool parameters are required');
    }

    return {
      ...builtTool,
      inputSchema: builtTool.parameters,
      onInputStart: 'onInputStart' in this.originalTool ? this.originalTool.onInputStart : undefined,
      onInputDelta: 'onInputDelta' in this.originalTool ? this.originalTool.onInputDelta : undefined,
      onInputAvailable: 'onInputAvailable' in this.originalTool ? this.originalTool.onInputAvailable : undefined,
    } as VercelToolV5;
  }

  build(): CoreTool {
    const providerTool = this.buildProviderTool(this.originalTool);
    if (providerTool) {
      return providerTool;
    }

    const definition = {
      type: 'function' as const,
      description: this.originalTool.description,
      parameters: this.getParameters(),
      outputSchema: this.getOutputSchema(),
      execute: this.originalTool.execute
        ? this.createExecute(
            this.originalTool,
            { ...this.options, description: this.originalTool.description },
            this.logType,
          )
        : undefined,
    };

    const model = this.options.model;

    const schemaCompatLayers = [];

    if (model) {
      let supportsStructuredOutputs = false;
      if (model.specificationVersion === 'v2') {
        supportsStructuredOutputs = true;
      } else {
        supportsStructuredOutputs = model.supportsStructuredOutputs ?? false;
      }

      const modelInfo = {
        modelId: model.modelId,
        supportsStructuredOutputs,
        provider: model.provider,
      };
      schemaCompatLayers.push(
        new OpenAIReasoningSchemaCompatLayer(modelInfo),
        new OpenAISchemaCompatLayer(modelInfo),
        new GoogleSchemaCompatLayer(modelInfo),
        new AnthropicSchemaCompatLayer(modelInfo),
        new DeepSeekSchemaCompatLayer(modelInfo),
        new MetaSchemaCompatLayer(modelInfo),
      );
    }

    const processedSchema = applyCompatLayer({
      schema: this.getParameters(),
      compatLayers: schemaCompatLayers,
      mode: 'aiSdkSchema',
    });

    let processedOutputSchema;

    if (this.getOutputSchema()) {
      processedOutputSchema = applyCompatLayer({
        schema: this.getOutputSchema(),
        compatLayers: schemaCompatLayers,
        mode: 'aiSdkSchema',
      });
    }

    return {
      ...definition,
      id: 'id' in this.originalTool ? this.originalTool.id : undefined,
      parameters: processedSchema,
      outputSchema: processedOutputSchema,
    };
  }
}
