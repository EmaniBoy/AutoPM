import "dotenv/config";
import express from "express";
import cors from "cors";
import { JIRA_AI_SYSTEM_PROMPT, buildUserPrompt } from "./prompts";
import { callNemotron } from "./nemotron";
import { JiraAIResponseSchema } from "./schema";
import { createEpicsAndStories } from "./jira";

const app = express();
app.use(cors());
app.use(express.json());

const PORT = Number(process.env.PORT || 4000);

// ---- Helper: parse + validate model JSON ----
function extractAndValidate(modelOutput: string) {
  let parsed: unknown;

  // Try direct parse
  try {
    parsed = JSON.parse(modelOutput);
  } catch {
    // Fallback: slice between first "{" and last "}"
    const start = modelOutput.indexOf("{");
    const end = modelOutput.lastIndexOf("}");
    if (start === -1 || end === -1 || end <= start) {
      throw new Error("Nemotron returned invalid JSON (no JSON object detected)");
    }
    const candidate = modelOutput.slice(start, end + 1);
    parsed = JSON.parse(candidate);
  }

  const result = JiraAIResponseSchema.safeParse(parsed);
  if (!result.success) {
    console.error("Schema validation failed: - server.ts:35", result.error.format());
    throw new Error("Nemotron JSON failed schema validation");
  }

  return result.data; // typed JiraAIResponse
}

// ---- 1) /jiraai/generate: only generate structured backlog ----
app.post("/jiraai/generate", async (req, res) => {
  const { projectKey, teamContext, rawInput } = req.body || {};

  if (!rawInput || typeof rawInput !== "string") {
    return res.status(400).json({ error: "rawInput is required" });
  }

  try {
    const userPrompt = buildUserPrompt(rawInput, projectKey, teamContext);

    const modelOutput = await callNemotron([
      { role: "system", content: JIRA_AI_SYSTEM_PROMPT },
      { role: "user", content: userPrompt },
    ]);

    const data = extractAndValidate(modelOutput);
    return res.json(data);
  } catch (err: any) {
    console.error("Error in /jiraai/generate: - server.ts:61", err);
    return res
      .status(502)
      .json({ error: err.message || "Nemotron generate failed" });
  }
});

// ---- 2) /jiraai/apply: take epics+stories JSON and create Jira issues ----
app.post("/jiraai/apply", async (req, res) => {
  const { projectKey, dryRun, ...rest } = req.body || {};

  try {
    const parsed = JiraAIResponseSchema.safeParse(rest);
    if (!parsed.success) {
      console.error("Invalid /apply payload: - server.ts:75", parsed.error.format());
      return res.status(400).json({
        error: "Invalid schema for epics/stories",
        details: parsed.error.format(),
      });
    }

    if (dryRun) {
      return res.json({
        message: "Dry run: no Jira issues created.",
        epics: parsed.data.epics,
        stories: parsed.data.stories,
      });
    }

    const result = await createEpicsAndStories(parsed.data, projectKey);

    return res.json({
      message: "Jira issues created successfully.",
      epics: result.epicKeyBySummary,
      stories: result.createdStories,
    });
  } catch (err: any) {
    console.error("Error in /jiraai/apply: - server.ts:98", err);
    return res.status(500).json({ error: err.message || "Jira apply failed" });
  }
});

// ---- 3) /jiraai/full-run: generate + create in one call ----
app.post("/jiraai/full-run", async (req, res) => {
  const { projectKey, teamContext, rawInput, dryRun } = req.body || {};

  if (!rawInput || typeof rawInput !== "string") {
    return res.status(400).json({ error: "rawInput is required" });
  }

  try {
    // Step 1: generate backlog with Nemotron
    const userPrompt = buildUserPrompt(rawInput, projectKey, teamContext);

    const modelOutput = await callNemotron([
      { role: "system", content: JIRA_AI_SYSTEM_PROMPT },
      { role: "user", content: userPrompt },
    ]);

    const generated = extractAndValidate(modelOutput);

    if (dryRun) {
      // Only show what WOULD be created
      return res.json({
        message: "Dry run: generated Jira backlog but did NOT create issues.",
        generated,
      });
    }

    // Step 2: create epics + stories in Jira
    const result = await createEpicsAndStories(generated, projectKey);

    return res.json({
      message: "Generated and created Jira issues successfully.",
      generated,
      epics: result.epicKeyBySummary,
      stories: result.createdStories,
    });
  } catch (err: any) {
    console.error("Error in /jiraai/fullrun: - server.ts:140", err);
    return res
      .status(500)
      .json({ error: err.message || "Full-run (generate+apply) failed" });
  }
});

// ---- Health ----
app.get("/", (_req, res) => {
  res.json({
    ok: true,
    service: "JiraAI (Nemotron -> Jira)",
    endpoints: ["/jiraai/generate", "/jiraai/apply", "/jiraai/full-run"],
  });
});

app.listen(PORT, () => {
  console.log(`JiraAI running at http://localhost:${PORT} - server.ts:157`);
});