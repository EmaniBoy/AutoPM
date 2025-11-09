import { z } from "zod";

// One Epic
export const EpicSchema = z.object({
  summary: z.string().min(3),
  description: z.string().min(3),
  priority: z.enum(["Highest", "High", "Medium", "Low"])
});

// One Story
export const StorySchema = z.object({
  summary: z.string().min(3),
  description: z.string().min(3), // "As a..., I want..., so that..."
  acceptance_criteria: z
    .array(z.string().min(5))
    .min(1)
    .max(10),
  priority: z.enum(["Highest", "High", "Medium", "Low"]),
  epic_summary: z.string().min(3) // must match an Epic.summary
});

// Full JiraAI response
export const JiraAIResponseSchema = z.object({
  epics: z.array(EpicSchema).min(1).max(5),
  stories: z.array(StorySchema).min(1)
});

export type Epic = z.infer<typeof EpicSchema>;
export type Story = z.infer<typeof StorySchema>;
export type JiraAIResponse = z.infer<typeof JiraAIResponseSchema>;
