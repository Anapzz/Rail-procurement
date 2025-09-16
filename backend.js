const express = require('express');
const cors = require('cors');
const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Example route
app.get('/', (req, res) => {
  res.send('Rail QR backend is running!');
});

// Generate QR and report endpoint
app.post('/generate_qr', (req, res) => {
  const { material_name, quantity, description, supplier } = req.body;
  // Generate a random fitting ID
  const fitting_id = 'FIT-' + Math.floor(Math.random() * 1000000);
  // Dummy QR code image URL (replace with real QR generation if needed)
  const qr_url = 'qr_img.png';
  res.json({
    fitting_id,
    material_name,
    quantity,
    description,
    supplier,
    qr_url
  });
});

// ...existing code...
app.listen(PORT, () => {
  console.log(`Backend server running on port ${PORT}`);
});