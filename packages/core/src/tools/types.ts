import type { ToolExecutionOptions, Tool, Schema } from 'ai';
import type { ToolCallOptions, Tool as ToolV5 } from 'ai-v5';
import type { JSONSchema7Type } from 'json-schema';
import type { ZodSchema, z } from 'zod';

import type { IAction, IExecutionContext, MastraUnion } from '../action';
import type { AITraceContext } from '../ai-tracing';
import type { Mastra } from '../mastra';
import type { RuntimeContext } from '../runtime-context';
import type { ToolStream } from './stream';

export type VercelTool = Tool;
export type VercelToolV5 = ToolV5;

export interface SubAgentStreamingConfig {
  enabled: boolean;
  depth?: 1 | 2;
  streamToolCalls?: boolean;
  streamText?: boolean;
  toolCallPrefix?: string;
  contextMetadata?: Record<string, any>;
}

// Define CoreTool as a discriminated union to match the AI SDK's Tool type
export type CoreTool = {
  id?: string;
  description?: string;
  parameters: ZodSchema | JSONSchema7Type | Schema;
  outputSchema?: ZodSchema | JSONSchema7Type | Schema;
  execute?: (params: any, options: ToolExecutionOptions) => Promise<any>;
} & (
  | {
      type?: 'function' | undefined;
      id?: string;
    }
  | {
      type: 'provider-defined';
      id: `${string}.${string}`;
      args: Record<string, unknown>;
    }
);

// Duplicate of CoreTool but with parameters as Schema to make it easier to work with internally
export type InternalCoreTool = {
  id?: string;
  description?: string;
  parameters: Schema;
  outputSchema?: Schema;
  execute?: (params: any, options: ToolExecutionOptions) => Promise<any>;
} & (
  | {
      type?: 'function' | undefined;
      id?: string;
    }
  | {
      type: 'provider-defined';
      id: `${string}.${string}`;
      args: Record<string, unknown>;
    }
);

export interface ToolExecutionContext<TSchemaIn extends z.ZodSchema | undefined = undefined>
  extends IExecutionContext<TSchemaIn> {
  mastra?: MastraUnion;
  runtimeContext: RuntimeContext;
  writer?: ToolStream<any>;
  aiTracingContext?: AITraceContext;
}

export interface ToolAction<
  TSchemaIn extends z.ZodSchema | undefined = undefined,
  TSchemaOut extends z.ZodSchema | undefined = undefined,
  TContext extends ToolExecutionContext<TSchemaIn> = ToolExecutionContext<TSchemaIn>,
> extends IAction<string, TSchemaIn, TSchemaOut, TContext, ToolExecutionOptions> {
  description: string;
  execute?: (
    context: TContext,
    options?: ToolExecutionOptions,
  ) => Promise<TSchemaOut extends z.ZodSchema ? z.infer<TSchemaOut> : unknown>;
  mastra?: Mastra;
  // Opt-in sub-agent streaming configuration (optional and additive)
  subAgentStreaming?: SubAgentStreamingConfig | ((args: { context: TContext }) => SubAgentStreamingConfig);
  onInputStart?: (options: ToolCallOptions) => void | PromiseLike<void>;
  onInputDelta?: (
    options: {
      inputTextDelta: string;
    } & ToolCallOptions,
  ) => void | PromiseLike<void>;
  onInputAvailable?: (
    options: {
      input: TSchemaIn extends z.ZodSchema ? z.infer<TSchemaIn> : unknown;
    } & ToolCallOptions,
  ) => void | PromiseLike<void>;
}
