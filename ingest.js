// ingest.js - Read PDFs, chunk text, embed with OpenAI, save to local Vectra index
const fs = require('fs');
const path = require('path');
const { PDFParse } = require('pdf-parse');
const { LocalIndex, OpenAIEmbeddings } = require('vectra');
require('dotenv').config();

const DATA_DIR = path.join(__dirname, 'data');
const INDEX_DIR = path.join(__dirname, 'vectorstore');

function cleanText(text) {
    return text
        .replace(/\r\n/g, '\n')
        .replace(/[ \t]+/g, ' ')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
}

function chunkText(text, chunkSize = 800, overlap = 150) {
    const chunks = [];
    let start = 0;
    while (start < text.length) {
        const end = Math.min(start + chunkSize, text.length);
        const chunk = text.slice(start, end).trim();
        if (chunk.length > 80) chunks.push(chunk);
        start += chunkSize - overlap;
    }
    return chunks;
}

async function ingestPDFs() {
    console.log('='.repeat(55));
    console.log('  Uraan RAG Ingestion Pipeline');
    console.log('='.repeat(55));

    if (!process.env.OPENAI_API_KEY) {
        console.error('OPENAI_API_KEY missing in .env'); process.exit(1);
    }

    // Create fresh index
    const index = new LocalIndex(INDEX_DIR);
    if (await index.isIndexCreated()) {
        await index.deleteIndex();
        console.log('Cleared existing index.');
    }
    await index.createIndex();

    const embeddings = new OpenAIEmbeddings({
        apiKey: process.env.OPENAI_API_KEY,
        model: 'text-embedding-3-small'
    });

    const pdfFiles = fs.readdirSync(DATA_DIR).filter(f => f.toLowerCase().endsWith('.pdf'));
    if (pdfFiles.length === 0) { console.error('No PDFs found in /data'); process.exit(1); }
    console.log(`Found ${pdfFiles.length} PDF(s): ${pdfFiles.join(', ')}\n`);

    let totalChunks = 0;

    for (const pdfFile of pdfFiles) {
        console.log(`Processing: ${pdfFile}`);
        const buffer = fs.readFileSync(path.join(DATA_DIR, pdfFile));
        const parser = new PDFParse({ data: buffer });
        const parsed = await parser.getText();
        const cleaned = cleanText(parsed.text);

        console.log(`  Pages: ${parsed.pages?.length ?? '?'} | Characters: ${cleaned.length}`);

        const chunks = chunkText(cleaned);
        console.log(`  Chunks: ${chunks.length}`);

        for (let i = 0; i < chunks.length; i++) {
            const result = await embeddings.createEmbeddings([chunks[i]]);
            const vector = result.output[0];
            await index.insertItem({
                vector,
                metadata: { source: pdfFile, chunk: i, text: chunks[i] }
            });
            if ((i + 1) % 20 === 0 || i === chunks.length - 1) {
                process.stdout.write(`  Embedded ${i + 1}/${chunks.length} chunks\r`);
            }
        }

        console.log(`\n  Done: ${pdfFile}\n`);
        totalChunks += chunks.length;
    }

    console.log('='.repeat(55));
    console.log(`Vector index saved to: ./vectorstore/`);
    console.log(`Total chunks embedded: ${totalChunks}`);
    console.log('Run "npm start" to launch the server.');
    console.log('='.repeat(55));
}

ingestPDFs().catch(err => {
    console.error('\nIngestion failed:', err.message);
    process.exit(1);
});
