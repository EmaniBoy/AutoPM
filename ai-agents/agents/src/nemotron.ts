import "dotenv/config";

const BASE_URL = process.env.NEMOTRON_API_BASE || "https://integrate.api.nvidia.com";
const API_KEY =
  process.env.NEMOTRON_API_KEY || // preferred
  process.env.NVIDIA_API_KEY ||   // fallback if you used this name
  "";
const MODEL = process.env.NEMOTRON_MODEL || "";

if (!API_KEY || !MODEL) {
  throw new Error(
    "Nemotron config missing in .env. Expected NEMOTRON_API_KEY (or NVIDIA_API_KEY) and NEMOTRON_MODEL."
  );
}

type Role = "system" | "user" | "assistant";

interface Message {
  role: Role;
  content: string;
}

export async function callNemotron(messages: Message[]): Promise<string> {
  const body = {
    model: MODEL,
    messages,
    temperature: 0,          // deterministic, avoids rambly thoughts
    max_tokens: 2048,
    // OpenAI-compatible hint: strongly request JSON object
    response_format: {
      type: "json_object"
    }
  };

  const res = await fetch(`${BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Nemotron request failed: ${res.status} - ${text}`);
  }

  const data = await res.json();
  const content = data?.choices?.[0]?.message?.content;

  if (!content || typeof content !== "string") {
    throw new Error("Nemotron returned empty or invalid content");
  }

  return content.trim();
}
