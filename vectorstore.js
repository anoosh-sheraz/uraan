// vectorstore.js - Pure JS vector store, no native dependencies
const fs = require('fs');
const path = require('path');

const STORE_FILE = path.join(__dirname, 'vectorstore.json');

function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function saveStore(items) {
    fs.writeFileSync(STORE_FILE, JSON.stringify(items));
}

function loadStore() {
    if (!fs.existsSync(STORE_FILE)) return [];
    return JSON.parse(fs.readFileSync(STORE_FILE, 'utf8'));
}

function query(queryVector, items, topK = 6, threshold = 0.4) {
    return items
        .map(item => ({ item, score: cosineSimilarity(queryVector, item.vector) }))
        .filter(r => r.score >= threshold)
        .sort((a, b) => b.score - a.score)
        .slice(0, topK);
}

module.exports = { saveStore, loadStore, query };
