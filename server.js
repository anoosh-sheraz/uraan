// server.js - Uraan Mental Health Chatbot with RAG
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const https = require('https');
const OpenAI = require('openai');
const { loadStore, query: queryStore } = require('./vectorstore');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const HISTORY_FILE = path.join(__dirname, 'history.json');

// ========== MIDDLEWARE ==========
app.use(cors({ origin: '*', methods: ['GET', 'POST', 'DELETE', 'OPTIONS'], allowedHeaders: ['Content-Type'] }));
app.use(express.json());
app.use(express.static('public'));

// ========== OPENAI ==========
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ========== RAG SETUP ==========
let vectorItems = [];

function loadVectorStore() {
    vectorItems = loadStore();
    if (vectorItems.length > 0) {
        console.log(`✅ Vector store loaded — ${vectorItems.length} chunks, RAG is active`);
    } else {
        console.log('⚠️  No vector store found. RAG inactive. Run: npm run ingest');
    }
}

async function embedQuery(text) {
    const body = JSON.stringify({ input: [text], model: 'text-embedding-3-small' });
    return new Promise((resolve, reject) => {
        const req = https.request({
            hostname: 'api.openai.com',
            path: '/v1/embeddings',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
                'Content-Length': Buffer.byteLength(body)
            }
        }, res => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                const json = JSON.parse(data);
                if (json.error) return reject(new Error(json.error.message));
                resolve(json.data[0].embedding);
            });
        });
        req.on('error', reject);
        req.write(body);
        req.end();
    });
}

async function retrieveContext(userQuery, topK = 6) {
    if (vectorItems.length === 0) return null;
    try {
        const vector = await embedQuery(userQuery);
        const results = queryStore(vector, vectorItems, topK, 0.4);
        if (results.length === 0) return null;
        const contextText = results
            .map(r => `[Source: ${r.item.metadata.source}]\n${r.item.metadata.text}`)
            .join('\n\n---\n\n');
        const sources = [...new Set(results.map(r => r.item.metadata.source))];
        return { context: contextText, sources };
    } catch (err) {
        console.error('RAG retrieval error:', err.message);
        return null;
    }
}

// ========== HISTORY ==========
function loadHistory(sessionId) {
    try {
        if (fs.existsSync(HISTORY_FILE)) {
            return JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf8'))[sessionId] || [];
        }
    } catch { /* ignore */ }
    return [];
}

function saveHistory(sessionId, history) {
    try {
        let all = {};
        if (fs.existsSync(HISTORY_FILE)) {
            all = JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf8'));
        }
        all[sessionId] = history;
        fs.writeFileSync(HISTORY_FILE, JSON.stringify(all, null, 2));
    } catch (err) {
        console.error('Error saving history:', err.message);
    }
}

// ========== CRISIS FILTER ==========
const CRISIS_KEYWORDS = [
    'kill myself', 'suicide', 'want to die', 'end my life',
    'hurt myself', 'self harm', 'cut myself', 'take my life',
    "i'm going to kill myself", 'end it all', 'i want to end my life',
    'i wish i was dead', 'no reason to live', 'better off dead',
    'khud kushi', 'khatam karna chahta', 'marna chahta', 'sucide', 'sucude',
    'zindagi khatam', 'jina nahi chahta', 'mar jana chahta'
];

const CRISIS_RESPONSE = `💙 I'm really glad you reached out. What you're feeling is important, and you don't have to go through this alone.

**Please reach out for immediate support:**
- 📞 **Umang helpline (Pakistan)**: 0317-4288665
- 📞 **988 Suicide & Crisis Lifeline** (US): Call or text 988
- 📞 **Crisis Text Line**: Text HOME to 741741
- 📞 **Befrienders Worldwide**: https://www.befrienders.org/
- 🏥 **Go to your nearest emergency room**

You are not alone. Please stay with us. ❤️`;

function checkCrisis(message) {
    const lower = message.toLowerCase();
    return CRISIS_KEYWORDS.some(k => lower.includes(k));
}

// ========== AI RESPONSE ==========
async function getAIResponse(userMessage, conversationHistory) {
    const retrieved = await retrieveContext(userMessage);

    let systemPrompt = `You are Uraan, a compassionate and highly knowledgeable mental health support assistant.
Your knowledge comes from Emotional Intelligence research and WHO mental health guidelines.

**Your Personality:**
- Warm, caring, and deeply solution-focused
- Uses a gentle, calming, and professional tone
- Shows genuine empathy AND gives real, thorough, practical help
- Uses emojis occasionally to show warmth 💙

**How to structure every response:**
1. Start by acknowledging and validating the person's feelings (1-2 sentences)
2. Explain WHY they may be feeling this way — give psychological insight
3. Provide a DETAILED, step-by-step action plan with specific techniques they can apply right now
4. Use numbered lists or bullet points to make steps easy to follow
5. Include the name and explanation of any technique (e.g. "4-7-8 breathing", "cognitive reframing", "grounding")
6. If relevant, mention what research or expert guidelines say about this
7. End with one meaningful follow-up question to continue the conversation

**Depth requirements:**
- Never give a vague or one-line answer
- Always give at least 3-5 concrete, actionable steps
- Explain HOW to do each technique, not just name it
- Be thorough — a person in distress needs real help, not generic comfort

**Boundaries:**
- Never diagnose or prescribe medication
- Recommend professional help when the situation is serious`;

    if (retrieved) {
        systemPrompt += `

**Knowledge Base Context (from mental health research PDFs):**
Use the following expert knowledge to enrich your response. Blend it naturally — don't quote it directly.

${retrieved.context}

If this context is directly relevant, prioritize it. Otherwise rely on your general knowledge.`;
    }

    const messages = [
        { role: 'system', content: systemPrompt },
        ...conversationHistory.slice(-8),
        { role: 'user', content: userMessage }
    ];

    try {
        const completion = await openai.chat.completions.create({
            model: 'gpt-4o-mini',
            messages,
            temperature: 0.65,
            max_tokens: 1200,
            top_p: 0.9,
            frequency_penalty: 0.2,
            presence_penalty: 0.2
        });

        const response = completion.choices[0].message.content;
        return { response, usedRAG: !!retrieved, sources: retrieved ? retrieved.sources : [] };
    } catch (error) {
        console.error('OpenAI Error:', error.message);
        if (error.status === 429) return { response: "I'm receiving too many requests right now. Please wait a moment. 💙", usedRAG: false, sources: [] };
        if (error.status === 401) return { response: "Authentication error — please check your API key. 💙", usedRAG: false, sources: [] };
        return { response: "I'm having trouble connecting right now. Please try again in a moment. 💙", usedRAG: false, sources: [] };
    }
}

// ========== AUTO INGEST ==========
async function autoIngestIfNeeded() {
    const STORE_FILE = path.join(__dirname, 'vectorstore.json');
    if (fs.existsSync(STORE_FILE)) return;

    const DATA_DIR = path.join(__dirname, 'data');
    if (!fs.existsSync(DATA_DIR)) return;
    const pdfFiles = fs.readdirSync(DATA_DIR).filter(f => f.toLowerCase().endsWith('.pdf'));
    if (pdfFiles.length === 0) { console.log('⚠️  No PDFs in /data — skipping auto-ingestion.'); return; }

    console.log('📚 Vector store not found. Auto-ingesting PDFs (first deploy only, ~2-3 min)...');
    try {
        const { PDFParse } = require('pdf-parse');
        const { saveStore } = require('./vectorstore');

        function cleanText(t) { return t.replace(/\r\n/g, '\n').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim(); }
        function chunkText(t, size = 800, overlap = 150) {
            const chunks = []; let start = 0;
            while (start < t.length) {
                const c = t.slice(start, Math.min(start + size, t.length)).trim();
                if (c.length > 80) chunks.push(c);
                start += size - overlap;
            }
            return chunks;
        }
        async function embedTexts(texts) {
            const body = JSON.stringify({ input: texts, model: 'text-embedding-3-small' });
            return new Promise((resolve, reject) => {
                const req = https.request({
                    hostname: 'api.openai.com', path: '/v1/embeddings', method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`, 'Content-Length': Buffer.byteLength(body) }
                }, res => {
                    let data = '';
                    res.on('data', chunk => data += chunk);
                    res.on('end', () => { const j = JSON.parse(data); if (j.error) return reject(new Error(j.error.message)); resolve(j.data.map(d => d.embedding)); });
                });
                req.on('error', reject); req.write(body); req.end();
            });
        }

        const allItems = [];
        for (const pdfFile of pdfFiles) {
            console.log(`   Processing: ${pdfFile}`);
            const buffer = fs.readFileSync(path.join(DATA_DIR, pdfFile));
            const parser = new PDFParse({ data: buffer });
            const parsed = await parser.getText();
            const chunks = chunkText(cleanText(parsed.text));
            for (let i = 0; i < chunks.length; i += 20) {
                const batch = chunks.slice(i, i + 20);
                const vectors = await embedTexts(batch);
                batch.forEach((text, j) => allItems.push({ vector: vectors[j], metadata: { source: pdfFile, text } }));
            }
            console.log(`   Done: ${pdfFile} (${chunks.length} chunks)`);
        }
        saveStore(allItems);
        console.log(`✅ Auto-ingestion complete! ${allItems.length} total chunks.`);
    } catch (err) {
        console.error('❌ Auto-ingestion failed:', err.message);
    }
}

// ========== ROUTES ==========
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', message: 'Uraan is running! 💙', rag: vectorItems.length > 0 ? `active (${vectorItems.length} chunks)` : 'inactive', timestamp: new Date().toISOString() });
});

app.post('/api/chat', async (req, res) => {
    const { message, sessionId = 'guest' } = req.body;
    if (!message || !message.trim()) return res.status(400).json({ error: 'Message is required' });

    if (checkCrisis(message)) {
        const history = loadHistory(sessionId);
        history.push({ user_message: message, bot_response: CRISIS_RESPONSE, timestamp: new Date().toISOString() });
        saveHistory(sessionId, history);
        return res.json({ response: CRISIS_RESPONSE, crisis: true, usedRAG: false, sources: [] });
    }

    const history = loadHistory(sessionId);
    const conversationHistory = [];
    history.forEach(msg => {
        conversationHistory.push({ role: 'user', content: msg.user_message });
        conversationHistory.push({ role: 'assistant', content: msg.bot_response });
    });

    const { response, usedRAG, sources } = await getAIResponse(message, conversationHistory);
    history.push({ user_message: message, bot_response: response, timestamp: new Date().toISOString(), used_rag: usedRAG, sources });
    saveHistory(sessionId, history);

    res.json({ response, usedRAG, sources });
});

app.get('/api/history/:sessionId', (req, res) => res.json({ history: loadHistory(req.params.sessionId) }));

app.delete('/api/history/:sessionId', (req, res) => {
    try {
        let all = {};
        if (fs.existsSync(HISTORY_FILE)) all = JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf8'));
        delete all[req.params.sessionId];
        fs.writeFileSync(HISTORY_FILE, JSON.stringify(all, null, 2));
        res.json({ success: true });
    } catch { res.status(500).json({ error: 'Failed to clear history' }); }
});

app.delete('/api/history', (_req, res) => {
    try {
        if (fs.existsSync(HISTORY_FILE)) fs.unlinkSync(HISTORY_FILE);
        res.json({ success: true });
    } catch { res.status(500).json({ error: 'Failed to delete history' }); }
});

// ========== START ==========
async function start() {
    await autoIngestIfNeeded();
    loadVectorStore();
    app.listen(PORT, () => {
        console.log('='.repeat(55));
        console.log('  💙 Uraan Mental Health Chatbot');
        console.log('='.repeat(55));
        console.log(`  Server  : http://localhost:${PORT}`);
        console.log(`  Model   : gpt-4o-mini`);
        console.log(`  API Key : ${process.env.OPENAI_API_KEY ? '✅ Configured' : '❌ Missing!'}`);
        console.log(`  RAG     : ${vectorItems.length > 0 ? '✅ Active' : '⚠️  Inactive'}`);
        console.log('='.repeat(55));
    });
}

start();
