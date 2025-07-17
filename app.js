// server.js
import express from 'express';
import cors from 'cors';
import { pipeline } from '@xenova/transformers';
import OpenAI from 'openai';

const app = express();
app.use(cors());
app.use(express.json());

let embedder;

const openrouter_api_key = ''; 
const openai = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: openrouter_api_key,
});

const init = async () => {
  console.log('Loading embedding model...');
  embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  console.log('Model ready');
};


app.post('/embed', async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: 'Missing text' });

    const output = await embedder(text, { pooling: 'mean', normalize: true });
    // Convert to plain array
    let embedding = output.data;
    if (Array.isArray(embedding[0])) {
      embedding = embedding[0];
    }
    embedding = Array.from(embedding);

    res.json(embedding);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Embedding generation failed' });
  }
});

app.post('/ncert-ai', async (req, res) => {
  try {
    const {  chapter } = req.body;
    // Fallbacks if not provided
    const prompt = `Generate a small para on NCERT class 11 physics chapter ${chapter },just return the para without any additional text.`;
    const chatCompletion = await openai.chat.completions.create({
      model: "deepseek/deepseek-r1-0528-qwen3-8b:free",
      messages: [{ role: "user", content: prompt }],
    });
    const aiPara = chatCompletion.choices[0].message.content;
    res.json({ para: aiPara });
  } catch (error) {
    console.error("Error communicating with OpenRouter:", error);
    res.status(500).json({ error: "AI generation failed" });
  }
});


init().then(() => {
  app.listen(3000, () => console.log('Embedding server running on port 3000'));
});
