# Nous Architecture — Inspired by Best Practices

## Philosophy

**Nous is built on lessons learned from:**
- OpenClaw/Claw-Code (harness architecture)
- b76 (FNC theory, persistent memory)
- Cognitive neuroscience (working memory, neural pathways)
- Industry best practices (Rust for systems, TS for UI)

**Not copied — inspired. Clean implementation from scratch.**

---

## Design Patterns (Inspired by)

### 1. MCP Protocol (from Claw-Code / OpenClaw)

**What they did well:**
- Standardized tool-calling interface
- Language-agnostic (any LLM can use)
- JSON-RPC based

**Our adaptation:**
- Same protocol, our implementation
- Nous-specific tools: `remember`, `recall`, `connect`, `get_working_memory`

```rust
// Inspired by MCP spec, our own code
#[derive(Serialize, Deserialize)]
struct NousTool {
    name: String,
    description: String,
    parameters: JsonSchema,
}

impl MCPTool for NousMemoryTool {
    fn execute(&self, params: Value) -> Result<Value, Error> {
        // Our implementation
    }
}
```

### 2. Plugin System (from Claw-Code)

**What they did well:**
- Modular architecture
- Hot-loading (optional)
- Clear interfaces

**Our adaptation:**
- FNC-native plugins (axons as active searchers)
- Memory plugins (different storage backends)
- Integration plugins (Claw-Code compatibility layer)

### 3. Session Management (from Claw-Code)

**What they did well:**
- Context preservation across calls
- State compaction
- Multiple concurrent sessions

**Our adaptation:**
- Session → Working Memory (7 slots)
- Persistent across restarts (unlike Claw-Code)
- Episodic tagging

### 4. Runtime Architecture (Rust best practices)

**Inspired by:**
- Tokio (async runtime)
- Actix (actor model)
- Claw-Code's rust/ structure

**Our structure:**
```
crates/
├── nouse-core/        # FNC implementation
├── nouse-mcp/         # MCP server (inspired protocol)
├── nouse-storage/     # Zulu DB wrapper
├── nouse-visual/      # Data for TS frontend
└── nouse-cli/         # CLI tool
```

### 5. Frontend Architecture (TypeScript/React patterns)

**Inspired by:**
- Three.js (3D visualization)
- D3.js (force-directed graphs)
- Next.js (modern web)

**Our approach:**
- WebGL for 3D node graph
- Real-time WebSocket updates
- Component-based UI

---

## What We Build Differently

| Aspect | Others | Nous |
|--------|--------|-------|
| **Memory** | Ephemeral | Persistent (FNC) |
| **Working Memory** | None | 7-slot system |
| **Knowledge** | Flat | Layered (surface → deep) |
| **Connections** | Static | Active "axons" |
| **Claims** | Stored as-is | Decomposed to MCD |

---

## Code Structure

### Rust Core (libnouse)

```rust
// Inspired by actor patterns, our own implementation
pub struct NousKernel {
    working_memory: WorkingMemory,  // 7 slots
    long_term_graph: Arc<RwLock<KnowledgeGraph>>,
    axon_pool: AxonPool,  // Active connection seekers
}

impl NousKernel {
    pub fn remember(&mut self, content: String, context: Context) -> NodeId {
        // Our FNC implementation
    }
    
    pub fn recall(&self, query: Query) -> Vec<Node> {
        // Our search algorithm
    }
}
```

### TypeScript Frontend

```typescript
// Inspired by reactive patterns, our own implementation
interface WorkingMemorySlot {
  id: string;
  node: KnowledgeNode;
  priority: number;
  lastAccessed: Date;
}

const WorkingMemoryView: React.FC = () => {
  const [slots, setSlots] = useState<WorkingMemorySlot[]>([]);
  // Our visualization logic
};
```

---

## Attribution & Respect

**Nous is possible because of:**
- OpenClaw/Claw-Code — showed MCP works for agents
- Anthropic — LLM research
- Cognitive scientists — working memory theory
- Rust community — systems programming done right

**We contribute back:**
- Open source under MIT
- Documentation of FNC theory
- Academic papers (The Larynx Problem)

---

## Testing Philosophy

**"Inspired by" means we test against the same problems:**
- Concurrent access (like Claw-Code)
- Memory safety (Rust guarantees)
- Performance (sub-10ms queries)
- But: Our own test cases, our own edge cases

---

*Built with respect for those who came before, 
but with unique vision for what comes next.*
