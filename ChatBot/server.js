const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");

const app = express();
const PORT = 3000;

app.use(cors());
app.use(bodyParser.json());

// MongoDB simulation (replace with real DB call if needed)
const mockDB = {
  USER_PROFILE: {
    name: "Yomna",
    skills: ["Python", "React", "C++"],
    education: "Tanta University"
  },
  USER_RESUME: {
    summary: "AI engineer with focus on LLMs",
    projects: ["Resume Builder", "Smart Health Assistant"]
  }
};

// Endpoint to receive request from FastAPI
app.post("/api/fetch-data", (req, res) => {
  const { needed_data, user_type, user_question , user_id} = req.body;

  console.log("✅ Data received from FastAPI:", req.body);

  // Prepare MongoDB data to return
  const result = {};
  needed_data.forEach((item) => {
    if (mockDB[item]) {
      result[item] = mockDB[item];
    }
  });

  return res.json({
    status: "ok",
    user_type,
    original_question: user_question,
    user_id,
    data: result
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Node.js backend running on http://localhost:${PORT}`);
});
