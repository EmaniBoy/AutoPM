import "dotenv/config";
import express from "express";
import { callNemotron } from "./nemotron";
import { JIRA_AI_SYSTEM_PROMPT, buildUserPrompt } from "./prompts";
import { JiraAIResponseSchema } from "./schema";
import { createEpicsAndStories } from "./jira";

const app = express();
app.use(express.json({ limit: "2mb" }));

// Health check
app.get("/", (_req, res) => {
  res.json({
    ok: true,
    service: "JiraAI",
    description:
      "Converts raw product input into Jira-ready Epics & Stories with acceptance criteria and priorities."
  });
});

// 1) Generate structured backlog from Nemotron
app.post("/jiraai/generate", async (req, res) => {
  try {
    const { projectKey, teamContext, rawInput } = req.body || {};

    if (!rawInput || typeof rawInput !== "string" || !rawInput.trim()) {
      return res
        .status(400)
        .json({ error: "rawInput (non-empty string) is required" });
    }

    const userPrompt = buildUserPrompt(rawInput, projectKey, teamContext);

    const modelOutput = await callNemotron([
      { role: "system", content: JIRA_AI_SYSTEM_PROMPT },
      { role: "user", content: userPrompt }
    ]);

    // Attempt to parse model output as JSON, with a fallback slice if it added extra text.
    let parsed: unknown;

    try {
      parsed = JSON.parse(modelOutput);
    } catch {
      const start = modelOutput.indexOf("{");
      const end = modelOutput.lastIndexOf("}");
      if (start !== -1 && end !== -1 && end > start) {
        const possibleJson = modelOutput.slice(start, end + 1);
        try {
          parsed = JSON.parse(possibleJson);
        } catch (e2) {
          console.error("Nemotron invalid JSON after slice: - server.ts:52", modelOutput);
          return res.status(502).json({
            error: "Nemotron returned invalid JSON",
            raw: modelOutput
          });
        }
      } else {
        console.error("Nemotron response had no JSON object: - server.ts:59", modelOutput);
        return res.status(502).json({
          error: "Nemotron returned invalid JSON",
          raw: modelOutput
        });
      }
    }

    // Validate against our strict schema
    const validated = JiraAIResponseSchema.parse(parsed);

    return res.status(200).json(validated);
  } catch (err: any) {
    console.error("Error in /jiraai/generate: - server.ts:72", err);
    return res
      .status(500)
      .json({ error: err.message || "Internal server error" });
  }
});

// 2) Apply generated backlog to Jira (optional)
app.post("/jiraai/apply", async (req, res) => {
  try {
    const { projectKey, epics, stories, dryRun } = req.body || {};

    const validated = JiraAIResponseSchema.parse({ epics, stories });

    if (dryRun) {
      return res.status(200).json({
        message: "Dry run: no Jira issues created.",
        epics: validated.epics,
        stories: validated.stories
      });
    }

    const result = await createEpicsAndStories(validated, projectKey);

    return res.status(200).json({
      message: "Jira issues created successfully.",
      epics: result.epicKeyBySummary,
      stories: result.createdStories
    });
  } catch (err: any) {
    console.error("Error in /jiraai/apply: - server.ts:102", err);
    return res
      .status(500)
      .json({ error: err.message || "Internal server error" });
  }
});

const PORT = Number(process.env.PORT) || 4000;
app.listen(PORT, () => {
  console.log(`JiraAI running at http://localhost:${PORT} - server.ts:111`);
});
