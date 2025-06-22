// personalizer.ts
import OpenAI from 'openai';
import * as readline from 'readline';
import * as dotenv from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';
dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const askMemoryConfirm = process.env.ASK_MEMORY_CONFIRM === 'true';

interface MemoryItem {
  id: string;
  key: string;
  value: string;
  text: string;
  category?: string;
  timestamp: string;
  embedding: number[];
}

interface ChatTurn {
  user: string;
  assistant: string;
}

const DATA_DIR = path.join(__dirname, '../data');
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  console.log("📂 Created data directory for memory and chat history.") ;
}

const MEMORY_FILE = path.join(DATA_DIR, 'memory.json');
const CHAT_HISTORY_FILE = path.join(DATA_DIR, 'chat_history.json');

let memoryStore: MemoryItem[] = [];
let chatHistory: ChatTurn[] = [];
const CHAT_HISTORY_LIMIT = 5;

function loadMemory() {
  if (fs.existsSync(MEMORY_FILE)) {
    const data = fs.readFileSync(MEMORY_FILE, 'utf-8');
    const items: MemoryItem[] = JSON.parse(data);
    memoryStore = items;
    console.log(`🔄 Loaded ${items.length} memory items.`);
  }
}

function saveMemory() {
  fs.writeFileSync(MEMORY_FILE, JSON.stringify(memoryStore, null, 2));
}

function clearMemory() {
  memoryStore = [];
  saveMemory();
  console.log("🧹 Cleared memory.");
}

function loadChatHistory() {
  if (fs.existsSync(CHAT_HISTORY_FILE)) {
    const data = fs.readFileSync(CHAT_HISTORY_FILE, 'utf-8');
    chatHistory = JSON.parse(data);
    console.log(`🗂️ Loaded ${chatHistory.length} chat history turns.`);
  }
}

function saveChatHistory() {
  fs.writeFileSync(CHAT_HISTORY_FILE, JSON.stringify(chatHistory.slice(-CHAT_HISTORY_LIMIT), null, 2));
}

function listChatHistory() {
  if (chatHistory.length === 0) {
    console.log("(no chat history stored)");
    return;
  }
  console.log("\n📜 Recent Chat History:");
  chatHistory.forEach((turn, i) => {
    console.log(`- [${i + 1}] User: ${turn.user}`);
    console.log(`         Assistant: ${turn.assistant}`);
  });
  console.log();
}

function clearChatHistory() {
  chatHistory = [];
  saveChatHistory();
  console.log("🧹 Cleared chat history.");
}

async function getEmbedding(text: string): Promise<number[]> {
  const res = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
  });
  return res.data[0].embedding;
}

function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val ** 2, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val ** 2, 0));
  return dot / (magA * magB);
}

function weightedAverage(vecA: number[], vecB: number[], weightA = 0.3, weightB = 0.7): number[] {
  return vecA.map((val, i) => val * weightA + vecB[i] * weightB);
}

function isDuplicateMemory(key: string, embedding: number[], threshold = 0.95): boolean {
  return memoryStore.some(mem =>
    mem.key === key && cosineSimilarity(mem.embedding, embedding) >= threshold
  );
}

async function addStructuredMemory(key: string, value: string, category?: string) {
  const keyEmbedding = await getEmbedding(key);
  const valueEmbedding = await getEmbedding(value);
  const embedding = weightedAverage(keyEmbedding, valueEmbedding);
  const text = `${key}: ${value}`;
  if (isDuplicateMemory(key, embedding)) {
    console.log(`⚠️ Skipped duplicate memory: [${key}] ${value}`);
    return;
  }

  const newMemory: MemoryItem = {
    id: (memoryStore.length + 1).toString(),
    key,
    value,
    text,
    category,
    timestamp: new Date().toISOString(),
    embedding,
  };
  memoryStore.push(newMemory);
  saveMemory();
  console.log(`🧠 Remembered: [${key}] ${value}${category ? ` (${category})` : ''}`);
}

async function autoExtractAndAddMemory(prompt: string, reply: string) {
  const extractPrompt = `From the following user message and the assistant's response, extract all useful memory facts about the user.\nRespond in JSON array format, each item must include a 'key' and 'value', and optionally a 'category'.\nRespond only with the JSON array. If nothing is relevant, respond with "[]".\n\nUser: ${prompt}\nAssistant: ${reply}`;

  const completion = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages: [
      { role: 'user', content: extractPrompt }
    ]
  });

  const raw = completion.choices[0].message?.content?.trim();
  if (raw && raw !== '[]') {
    const cleaned = raw.replace(/^```(?:json)?|```$/g, '').trim();
    try {
      const items = JSON.parse(cleaned);
      if (Array.isArray(items)) {
        for (const item of items) {
          if (item.key && item.value) {
            if (askMemoryConfirm) {
              const confirmed = await askUserToConfirm(item.key, item.value);
              if (confirmed) {
                await addStructuredMemory(item.key, item.value, item.category);
              }
            } else {
              await addStructuredMemory(item.key, item.value, item.category);
            }
          }
        }
      }
    } catch (err) {
      console.warn('⚠️ Failed to parse memory JSON array:', cleaned);
    }
  }
}

async function askUserToConfirm(key: string, value: string): Promise<boolean> {
  return new Promise(resolve => {
    rl.question(`💾 Save memory [${key}: ${value}]? (y/n) `, answer => {
      resolve(answer.toLowerCase().startsWith('y'));
    });
  });
}

function deleteMemoryById(id: string) {
  const index = memoryStore.findIndex(mem => mem.id === id);
  if (index >= 0) {
    const removed = memoryStore.splice(index, 1);
    saveMemory();
    console.log(`❌ Deleted memory: "${removed[0].text}"`);
  } else {
    console.log(`⚠️ No memory found with ID ${id}`);
  }
}

function listMemories(categoryFilter?: string) {
  const memories = categoryFilter
    ? memoryStore.filter(m => m.category === categoryFilter)
    : memoryStore;

  if (memories.length === 0) {
    console.log("(no memories stored)");
    return;
  }

  console.log(`\n🧾 Memory Items${categoryFilter ? ` (category: ${categoryFilter})` : ''}:`);
  memories.forEach(m => {
    const categoryInfo = m.category ? ` [${m.category}]` : '';
    console.log(`- [${m.id}] ${m.key}: ${m.value}${categoryInfo} (${m.timestamp})`);
  });
  console.log();
}

async function retrieveRelevantMemories(query: string, topN = 3): Promise<MemoryItem[]> {
  const queryEmbedding = await getEmbedding(query);
  return memoryStore
    .map(mem => ({
      ...mem,
      score: cosineSimilarity(queryEmbedding, mem.embedding),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topN);
}

async function respondWithMemory(query: string) {
  //console.log(`\n💬 User: ${query}\n`);
  const relevant = await retrieveRelevantMemories(query);
  const memoryContext = relevant.map(m => `- ${m.key}: ${m.value}`).join('\n');

  const historyContext = chatHistory
    .map(turn => `User: ${turn.user}\nAssistant: ${turn.assistant}`)
    .join('\n');

  const fullPrompt = `You are a helpful assistant.\n\nUser's persistent memory:\n${memoryContext}\n\nRecent conversation:\n${historyContext}\n\nUser: ${query}`;

  const res = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: fullPrompt }],
  });

  const reply = res.choices[0].message?.content;
  console.log(`\n🤖 ${reply}\n`);

  if (reply) {
    chatHistory.push({ user: query, assistant: reply });
    if (chatHistory.length > CHAT_HISTORY_LIMIT) {
      chatHistory = chatHistory.slice(-CHAT_HISTORY_LIMIT);
    }
    saveChatHistory();
    await autoExtractAndAddMemory(query, reply);
  }
}

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

function ask(question: string): Promise<string> {
  return new Promise(resolve => rl.question(question, resolve));
}

function listCategories() {
  const categoryCounts = memoryStore.reduce<Record<string, number>>((acc, m) => {
    if (m.category) {
      acc[m.category] = (acc[m.category] || 0) + 1;
    }
    return acc;
  }, {});

  const categories = Object.entries(categoryCounts);

  if (categories.length === 0) {
    console.log("(no categories found)");
    return;
  }
  console.log("🏷️ Categories in memory:");
  categories.forEach(([cat, count]) => console.log(`- ${cat} (${count})`));
  console.log();
}

async function main() {
  console.log("\n🧠 Personal Assistant with Structured Memory\n---------------------------------------------");
  loadMemory();
  loadChatHistory();

  while (true) {
    const input = await ask("[Categories/List/Delete/History/Clear/Exit] or just ask a question > ");
    const lower = input.toLowerCase();
    if (lower === "categories") {
      listCategories();
    } else if (lower === "list") {
      listMemories();
    } else if (lower.startsWith("list ")) {
      const category = input.slice(5).trim();
      listMemories(category);
    } else if (lower === "history") {
      listChatHistory();
    } else if (lower === "clear") {
      clearChatHistory();
    } else if (lower.startsWith("delete all")) {
      clearMemory();
    } else if (lower.startsWith("delete ")) {
      deleteMemoryById(input.slice(7).trim());
    } else if (lower === "exit") {
      break;
    } else {
      await respondWithMemory(input);
    }
  }
  rl.close();
}

main();
