export default function handler(req, res) {
  res.status(200).json({
    status: "ok",
    project: "NeuronGen SexyScripts",
    supabase: process.env.SUPABASE_URL ? "connected" : "missing",
    timestamp: new Date().toISOString()
  });
}
