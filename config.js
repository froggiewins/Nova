// Nova's trainer database (Supabase): visitors' lessons ("Teach Nova") and the titles of
// Wikipedia articles Nova found, so he can learn them in his nightly training.
// This is the PUBLISHABLE key, which is meant to be public: the database rules only let it
// ADD rows, never read, change or delete them.
// NEVER put the secret key (sb_secret_...) or service_role key here.
window.NOVA_CONFIG = {
  SUPABASE_URL: "https://wpnhgaykpigeqqdkqobn.supabase.co",
  SUPABASE_ANON_KEY: "sb_publishable_S75LLKNzaKhBwRuVJSRSGg_jpuq5l-e",
};
