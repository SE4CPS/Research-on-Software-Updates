require("dotenv").config();

const express = require("express");
const cors = require("cors");
const path = require("path");

const queryRouter = require("./routes/query");
const { loadComponentNames } = require("./services/componentNames");

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

app.use("/api/query", queryRouter);

app.get("/api/health", (req, res) => {
  res.json({ status: "ok", uptime: process.uptime(), timestamp: new Date().toISOString() });
});

if (process.env.NODE_ENV === "production") {
  const distPath = path.join(__dirname, "..", "dist");
  app.use(express.static(distPath));
  app.get("*", (req, res) => {
    res.sendFile(path.join(distPath, "index.html"));
  });
}

app.listen(PORT, () => {
  console.log(`\n  GenAI Abstain App`);
  console.log(`  http://localhost:${PORT}`);
  console.log(`  ENV: ${process.env.NODE_ENV || "development"}\n`);
  // Load component names in the background — server is ready immediately,
  // first few requests fall back to built-in patterns if list hasn't loaded yet.
  loadComponentNames();
});