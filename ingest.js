// ingest.js - Read PDFs, embed with OpenAI, save to plain JSON file
const fs = require('fs');
const path = require('path');
const https = require('https');
const { PDFParse } = require('pdf-parse');
const { saveStore } = require('./vectorstore');
require('dotenv').config();

const DATA_DIR = path.join(__dirname, 'data');

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

async function embedTexts(texts) {
    const body = JSON.stringify({ input: texts, model: 'text-embedding-3-small' });
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
                resolve(json.data.map(d => d.embedding));
            });
        });
        req.on('error', reject);
        req.write(body);
        req.end();
    });
}

async function ingestPDFs() {
    console.log('='.repeat(55));
    console.log('  Uraan RAG Ingestion Pipeline');
    console.log('='.repeat(55));

    if (!process.env.OPENAI_API_KEY) { console.error('OPENAI_API_KEY missing'); process.exit(1); }

    const pdfFiles = fs.readdirSync(DATA_DIR).filter(f => f.toLowerCase().endsWith('.pdf'));
    if (pdfFiles.length === 0) { console.error('No PDFs in /data'); process.exit(1); }
    console.log(`Found ${pdfFiles.length} PDF(s): ${pdfFiles.join(', ')}\n`);

    const allItems = [];

    for (const pdfFile of pdfFiles) {
        console.log(`Processing: ${pdfFile}`);
        const buffer = fs.readFileSync(path.join(DATA_DIR, pdfFile));
        const parser = new PDFParse({ data: buffer });
        const parsed = await parser.getText();
        const chunks = chunkText(cleanText(parsed.text));
        console.log(`  Chunks: ${chunks.length}`);

        // Embed in batches of 20
        const batchSize = 20;
        for (let i = 0; i < chunks.length; i += batchSize) {
            const batch = chunks.slice(i, i + batchSize);
            const vectors = await embedTexts(batch);
            batch.forEach((text, j) => {
                allItems.push({ vector: vectors[j], metadata: { source: pdfFile, text } });
            });
            process.stdout.write(`  Embedded ${Math.min(i + batchSize, chunks.length)}/${chunks.length}\r`);
        }
        console.log(`\n  Done: ${pdfFile}\n`);
    }

    saveStore(allItems);
    console.log('='.repeat(55));
    console.log(`Total chunks: ${allItems.length}`);
    console.log('Saved to vectorstore.json');
    console.log('Run "npm start" to launch.');
    console.log('='.repeat(55));
}

ingestPDFs().catch(err => { console.error('Failed:', err.message); process.exit(1); });
