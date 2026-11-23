// Where visitors' lessons ("Teach Nova") are sent.
// Leave these empty to hide the Teach button.
// Use your Supabase project's URL and its "anon public" key. The anon key is meant to be public:
// the database rules only let it ADD lessons, never read or change them.
// NEVER put the service_role key here.
window.NOVA_CONFIG = {
  SUPABASE_URL: "",
  SUPABASE_ANON_KEY: "",
};
