// server.js
import express from 'express';
import cors from 'cors';
import { pipeline } from '@xenova/transformers';
import { YoutubeTranscript } from 'youtube-transcript';

const app = express();
app.use(cors());
app.use(express.json());

let embedder;

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
    embedding = Array.from(embedding); // <-- This ensures it's a real array

    res.json(embedding);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Embedding generation failed' });
  }
});




init().then(() => {
  app.listen(3000, () => console.log('Embedding server running on port 3000'));
});
