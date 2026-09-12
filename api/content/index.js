import { createClient } from "@supabase/supabase-js";

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") return res.status(200).end();

  if (req.method === "GET") {
    const { data, error } = await supabase
      .from("content")
      .select("*")
      .eq("is_public", true)
      .order("created_at", { ascending: false });

    if (error) return res.status(500).json({ error: error.message });
    return res.status(200).json({ data });
  }

  if (req.method === "POST") {
    const { title, description, image_url, is_public, user_id } = req.body;
    if (!title) return res.status(400).json({ error: "title is required" });

    const { data, error } = await supabase
      .from("content")
      .insert([{ title, description, image_url, is_public: !!is_public, user_id }])
      .select();

    if (error) return res.status(500).json({ error: error.message });
    return res.status(201).json({ data });
  }

  return res.status(405).json({ error: "Method not allowed" });
}
