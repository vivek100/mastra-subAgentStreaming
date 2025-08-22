import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import type { WritableStream } from 'stream/web';
import { CoreToolBuilder } from './tool-builder/builder';
import type { ToolAction } from './types';

// Helper to create a capturable writable stream
function createCaptureStream(chunks: any[]): WritableStream<any> {
  return new WritableStream<any>({
    write(chunk) {
      chunks.push(chunk);
    },
  });
}

function createSubAgentStream(parts: any[]): any {
  // Returns an object mimicking MastraModelOutput with fullStream
  const fullStream = new ReadableStream<any>({
    start(controller) {
      for (const p of parts) controller.enqueue(p);
      controller.close();
    },
  });
  return {
    fullStream,
    text: Promise.resolve(parts
      .filter(p => p.type === 'text-delta')
      .map((p: any) => p.payload.text)
      .join('')),
  };
}

describe('Sub-agent streaming (opt-in)', () => {
  it('forwards sub-agent tool calls and text with prefix', async () => {
    const captured: any[] = [];
    const writable = createCaptureStream(captured);

    // Stub agent returned by mastra.getAgent
    const stubAgent = {
      name: 'calc-agent',
      streamVNext: async (_prompt: string) =>
        createSubAgentStream([
          { type: 'tool-call', payload: { toolCallId: 'tc1', toolName: 'derivative', args: { fn: 'x^2' } } },
          { type: 'text-delta', payload: { text: 'Computing derivative...' } },
          { type: 'tool-result', payload: { toolCallId: 'tc1', toolName: 'derivative', result: { derivative: '2x' } } },
        ]),
    };

    const mastraStub = {
      getAgent: () => stubAgent,
    } as any;

    const tool: ToolAction<any, any, any> = {
      id: 'ask_calculus',
      description: 'Delegate calculus problems',
      inputSchema: z.object({ prompt: z.string() }),
      execute: async ({ context, mastra }) => {
        const a = mastra.getAgent('calc-agent');
        const s = await a.streamVNext(context.prompt);
        // Consume to ensure forwarding executes
        await s.text;
        return { text: 'done' } as any;
      },
      subAgentStreaming: { enabled: true, streamToolCalls: true, streamText: true, depth: 1, toolCallPrefix: 'calc' },
      mastra: mastraStub as any,
    } as any;

    const built = new CoreToolBuilder({ originalTool: tool, options: {
      name: 'ask_calculus',
      runId: 'run-parent',
      agentName: 'orchestrator',
      writableStream: writable,
      runtimeContext: undefined as any,
    }, logType: 'tool' }).build();

    expect(typeof built.execute).toBe('function');

    await built.execute?.({ prompt: 'd/dx x^2' }, { toolCallId: 'call-1' } as any);

    // Assertions
    const types = captured.map(c => c.type);
    expect(types).toContain('sub-agent-start');
    expect(types).toContain('sub-tool-call');
    expect(types).toContain('sub-tool-result');
    expect(types).toContain('sub-text');
    expect(types).toContain('sub-agent-end');

    const subCall = captured.find(c => c.type === 'sub-tool-call');
    expect(subCall.payload.toolName).toBe('calc.derivative');

    const subResult = captured.find(c => c.type === 'sub-tool-result');
    expect(subResult.payload.toolName).toBe('calc.derivative');
    expect(subResult.payload.result).toEqual({ derivative: '2x' });
  });

  it('does nothing when streaming disabled', async () => {
    const captured: any[] = [];
    const writable = createCaptureStream(captured);

    const stubAgent = {
      name: 'calc-agent',
      streamVNext: async (_prompt: string) =>
        createSubAgentStream([
          { type: 'tool-call', payload: { toolCallId: 'tc1', toolName: 'derivative', args: { fn: 'x^2' } } },
          { type: 'text-delta', payload: { text: 'Computing derivative...' } },
          { type: 'tool-result', payload: { toolCallId: 'tc1', toolName: 'derivative', result: { derivative: '2x' } } },
        ]),
    };

    const mastraStub = { getAgent: () => stubAgent } as any;

    const tool: ToolAction<any, any, any> = {
      id: 'ask_calculus',
      description: 'Delegate calculus problems',
      inputSchema: z.object({ prompt: z.string() }),
      execute: async ({ context, mastra }) => {
        const a = mastra.getAgent('calc-agent');
        const s = await a.streamVNext(context.prompt);
        await s.text;
        return { text: 'done' } as any;
      },
      // subAgentStreaming omitted (disabled)
      mastra: mastraStub as any,
    } as any;

    const built = new CoreToolBuilder({ originalTool: tool, options: {
      name: 'ask_calculus',
      runId: 'run-parent',
      agentName: 'orchestrator',
      writableStream: writable,
      runtimeContext: undefined as any,
    }, logType: 'tool' }).build();

    await built.execute?.({ prompt: 'd/dx x^2' }, { toolCallId: 'call-1' } as any);

    // No sub-* events should be emitted
    expect(captured.every(c => !String(c.type).startsWith('sub-'))).toBe(true);
  });
});
