// server.js - Uraan Mental Health Chatbot with RAG (PDF Knowledge Base)
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const OpenAI = require('openai');
const vectra = require('vectra');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const VECTOR_STORE_DIR = path.join(__dirname, 'vectorstore');
const HISTORY_FILE = path.join(__dirname, 'history.json');

// ========== MIDDLEWARE ==========
app.use(cors({
    origin: '*',
    methods: ['GET', 'POST', 'DELETE', 'PUT', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.json());
app.use(express.static('public'));

// ========== OPENAI SETUP ==========
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ========== RAG SETUP ==========
let ragIndex = null;
let ragEmbeddings = null;

async function loadVectorStore() {
    if (!fs.existsSync(VECTOR_STORE_DIR)) {
        console.log('⚠️  No vector store found. Run: node ingest.js');
        return;
    }
    try {
        const index = new vectra.LocalIndex(VECTOR_STORE_DIR);
        if (!(await index.isIndexCreated())) {
            console.log('⚠️  Vector store folder exists but index is missing. Run: node ingest.js');
            return;
        }
        ragIndex = index;
        ragEmbeddings = new vectra.OpenAIEmbeddings({
            apiKey: process.env.OPENAI_API_KEY,
            model: 'text-embedding-3-small'
        });
        console.log('✅ Vector store loaded — RAG is active');
    } catch (err) {
        console.error('❌ Failed to load vector store:', err.message);
    }
}

// Retrieve the most relevant chunks from the PDF knowledge base
async function retrieveContext(query, topK = 6) {
    if (!ragIndex || !ragEmbeddings) return null;
    try {
        const result = await ragEmbeddings.createEmbeddings([query]);
        const vector = result.output[0];
        const results = await ragIndex.queryItems(vector, topK);

        // score is cosine similarity (0–1); keep only strong matches
        const relevant = results.filter(r => r.score > 0.4);
        if (relevant.length === 0) return null;

        const contextText = relevant
            .map(r => `[Source: ${r.item.metadata.source}]\n${r.item.metadata.text}`)
            .join('\n\n---\n\n');

        const sources = [...new Set(relevant.map(r => r.item.metadata.source))];
        return { context: contextText, sources };
    } catch (err) {
        console.error('RAG retrieval error:', err.message);
        return null;
    }
}

// ========== JSON FILE STORAGE ==========
function loadHistory(sessionId) {
    try {
        if (fs.existsSync(HISTORY_FILE)) {
            const data = fs.readFileSync(HISTORY_FILE, 'utf8');
            return JSON.parse(data)[sessionId] || [];
        }
        return [];
    } catch {
        return [];
    }
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

// ========== CRISIS SAFETY FILTER ==========
const CRISIS_KEYWORDS = [
    'kill myself', 'suicide', 'want to die', 'end my life',
    'hurt myself', 'self harm', 'cut myself', 'take my life',
    "i'm going to kill myself", 'end it all', 'i want to end my life',
    'i wish i was dead', 'no reason to live', 'better off dead',
    // Urdu
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

// ========== MAIN AI RESPONSE (RAG-ENHANCED) ==========
async function getAIResponse(userMessage, conversationHistory) {
    // Try to retrieve relevant PDF context
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
- If the knowledge base has relevant information, use it to give expert-backed detail
- Be thorough — a person in distress needs real help, not generic comfort

**Boundaries:**
- Never diagnose or prescribe medication
- Recommend professional help when the situation is serious
- You are NOT a replacement for professional therapy — say so when appropriate`;

    // Inject PDF knowledge if relevant context was found
    if (retrieved) {
        systemPrompt += `

**Knowledge Base Context (from mental health research PDFs):**
Use the following expert knowledge to inform and enrich your response. You don't need to quote it directly — absorb it and respond naturally, with depth and accuracy. Blend it with your own understanding.

${retrieved.context}

If this context is directly relevant, prioritize it. If not, rely on your general knowledge.`;
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
        const usedRAG = !!retrieved;
        const sources = retrieved ? retrieved.sources : [];

        return { response, usedRAG, sources };
    } catch (error) {
        console.error('OpenAI Error:', error.message);
        if (error.status === 429) return { response: "I'm receiving too many requests right now. Please wait a moment and try again. 💙", usedRAG: false, sources: [] };
        if (error.status === 401) return { response: "I'm having trouble authenticating. Please check your API key. 💙", usedRAG: false, sources: [] };
        return { response: "I'm having trouble connecting right now. Please try again in a moment. 💙", usedRAG: false, sources: [] };
    }
}

// ========== API ROUTES ==========

app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        message: 'Uraan is running! 💙',
        rag: ragIndex ? 'active' : 'not loaded (run node ingest.js)',
        timestamp: new Date().toISOString()
    });
});

app.post('/api/chat', async (req, res) => {
    const { message, sessionId = 'guest' } = req.body;

    if (!message || !message.trim()) {
        return res.status(400).json({ error: 'Message is required' });
    }

    // Crisis check first — always
    if (checkCrisis(message)) {
        const history = loadHistory(sessionId);
        history.push({
            user_message: message,
            bot_response: CRISIS_RESPONSE,
            timestamp: new Date().toISOString()
        });
        saveHistory(sessionId, history);
        return res.json({ response: CRISIS_RESPONSE, crisis: true, usedRAG: false, sources: [] });
    }

    // Build conversation context for OpenAI
    const history = loadHistory(sessionId);
    const conversationHistory = [];
    history.forEach(msg => {
        conversationHistory.push({ role: 'user', content: msg.user_message });
        conversationHistory.push({ role: 'assistant', content: msg.bot_response });
    });

    // Get RAG-enhanced AI response
    const { response, usedRAG, sources } = await getAIResponse(message, conversationHistory);

    // Save to history
    history.push({
        user_message: message,
        bot_response: response,
        timestamp: new Date().toISOString(),
        used_rag: usedRAG,
        sources
    });
    saveHistory(sessionId, history);

    res.json({ response, usedRAG, sources });
});

app.get('/api/history/:sessionId', (req, res) => {
    const history = loadHistory(req.params.sessionId);
    res.json({ history });
});

app.delete('/api/history/:sessionId', (req, res) => {
    try {
        let all = {};
        if (fs.existsSync(HISTORY_FILE)) {
            all = JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf8'));
        }
        delete all[req.params.sessionId];
        fs.writeFileSync(HISTORY_FILE, JSON.stringify(all, null, 2));
        res.json({ success: true });
    } catch {
        res.status(500).json({ error: 'Failed to clear history' });
    }
});

app.delete('/api/history', (_req, res) => {
    try {
        if (fs.existsSync(HISTORY_FILE)) fs.unlinkSync(HISTORY_FILE);
        res.json({ success: true });
    } catch {
        res.status(500).json({ error: 'Failed to delete history' });
    }
});

// ========== AUTO INGEST IF VECTOR STORE MISSING ==========
async function autoIngestIfNeeded() {
    if (fs.existsSync(VECTOR_STORE_DIR)) return; // already exists

    const DATA_DIR = path.join(__dirname, 'data');
    const pdfFiles = fs.existsSync(DATA_DIR)
        ? fs.readdirSync(DATA_DIR).filter(f => f.toLowerCase().endsWith('.pdf'))
        : [];

    if (pdfFiles.length === 0) {
        console.log('⚠️  No PDFs found in /data — skipping auto-ingestion.');
        return;
    }

    console.log('📚 Vector store not found. Auto-ingesting PDFs...');
    console.log('   This runs once and may take 2-3 minutes on first deploy.');

    try {
        const { PDFParse } = require('pdf-parse');
        const { LocalIndex, OpenAIEmbeddings: VectraEmbeddings } = require('vectra');

        function cleanText(text) {
            return text.replace(/\r\n/g, '\n').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
        }
        function chunkText(text, size = 800, overlap = 150) {
            const chunks = [];
            let start = 0;
            while (start < text.length) {
                const chunk = text.slice(start, Math.min(start + size, text.length)).trim();
                if (chunk.length > 80) chunks.push(chunk);
                start += size - overlap;
            }
            return chunks;
        }

        const index = new LocalIndex(VECTOR_STORE_DIR);
        await index.createIndex();
        const embeddings = new VectraEmbeddings({
            apiKey: process.env.OPENAI_API_KEY,
            model: 'text-embedding-3-small'
        });

        for (const pdfFile of pdfFiles) {
            console.log(`   Processing: ${pdfFile}`);
            const buffer = fs.readFileSync(path.join(DATA_DIR, pdfFile));
            const parser = new PDFParse({ data: buffer });
            const parsed = await parser.getText();
            const chunks = chunkText(cleanText(parsed.text));
            for (let i = 0; i < chunks.length; i++) {
                const result = await embeddings.createEmbeddings([chunks[i]]);
                await index.insertItem({
                    vector: result.output[0],
                    metadata: { source: pdfFile, chunk: i, text: chunks[i] }
                });
            }
            console.log(`   Done: ${pdfFile} (${chunks.length} chunks)`);
        }
        console.log('✅ Auto-ingestion complete!');
    } catch (err) {
        console.error('❌ Auto-ingestion failed:', err.message);
    }
}

// ========== START SERVER ==========
async function start() {
    await autoIngestIfNeeded();
    await loadVectorStore();

    app.listen(PORT, () => {
        console.log('='.repeat(55));
        console.log('  💙 Uraan Mental Health Chatbot');
        console.log('='.repeat(55));
        console.log(`  Server  : http://localhost:${PORT}`);
        console.log(`  Model   : gpt-4o-mini`);
        console.log(`  API Key : ${process.env.OPENAI_API_KEY ? '✅ Configured' : '❌ Missing!'}`);
        console.log(`  RAG     : ${ragIndex ? '✅ Active (PDF knowledge loaded)' : '⚠️  Inactive — run: node ingest.js'}`);
        console.log(`  Storage : history.json`);
        console.log('='.repeat(55));
    });
}

start();
