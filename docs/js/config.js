// Point this at your Supabase project + public Storage bucket. Find the
// Project URL under Settings -> API in the Supabase dashboard.
// Example: "https://abcdefghijklmno.supabase.co"
const SUPABASE_URL = 'https://wqapdlssacjfkiobbsxx.supabase.co';
const SUPABASE_BUCKET = 'lstm-data';

// Builds the public Storage URL for a file, e.g. assetUrl('data/history.json').
// Works with no auth/key because the bucket is public — anyone can read,
// nobody can write without the (never-shipped) service role key.
function assetUrl(path) {
  return `${SUPABASE_URL}/storage/v1/object/public/${SUPABASE_BUCKET}/${path}`;
}
