-- ============================================================
-- Migration: create_content_table
-- Project:   NeuronGen SexyScripts
-- Target:    siinmfgkjpysehzuovyy
-- ============================================================

-- 1. Таблица контента
CREATE TABLE IF NOT EXISTS content (
  id           UUID                     DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id      UUID                     REFERENCES auth.users(id) ON DELETE CASCADE,
  title        TEXT                     NOT NULL,
  description  TEXT,
  image_url    TEXT,
  is_public    BOOLEAN                  DEFAULT false,
  created_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Индексы для производительности
CREATE INDEX IF NOT EXISTS idx_content_user   ON content(user_id);
CREATE INDEX IF NOT EXISTS idx_content_public ON content(is_public) WHERE is_public = true;

-- 3. Функция автообновления updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger
DROP TRIGGER IF EXISTS content_updated_at ON content;
CREATE TRIGGER content_updated_at
  BEFORE UPDATE ON content
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- Row-Level Security (RLS)
-- ============================================================
ALTER TABLE content ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users see own posts"     ON content FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Public posts viewable"   ON content FOR SELECT USING (is_public = true);
CREATE POLICY "Owners insert own posts" ON content FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Owners update own posts" ON content FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Owners delete own posts" ON content FOR DELETE USING (auth.uid() = user_id);
