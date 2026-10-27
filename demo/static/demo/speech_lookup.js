// Speech -> Sign word lookup: English text -> list of BSL clips to play.
// Order of checks for each word: phrase (up to 3 words) -> word -> BSLDict alternative ->
// irregular form (went -> go) -> word root (helping -> help) -> WordNet synonym -> fingerspelling.

const CONTRACTIONS = {
  "don't": 'do not', "doesn't": 'does not', "didn't": 'did not', "can't": 'cannot', "cannot": 'cannot',
  "won't": 'will not', "isn't": 'is not', "aren't": 'are not', "wasn't": 'was not', "i'm": 'i am',
  "you're": 'you are', "we're": 'we are', "they're": 'they are', "it's": 'it is', "that's": 'that is',
  "i've": 'i have', "i'll": 'i will', "i'd": 'i would', "let's": 'let us', "what's": 'what is',
  "where's": 'where is', "how's": 'how is', "there's": 'there is',
};

// Grammar words that BSL usually leaves out. They are dropped only when no sign exists for them.
const SKIP = new Set(['a', 'an', 'the', 'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being', 'to', 'of',
  'and', 'or', 'but', 'it', 'its', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'could', 'has',
  'have', 'had', 'that', 'this', 'these', 'those', 'at', 'in', 'on', 'for', 'with', 'by', 'from', 'as',
  'so', 'very', 'just', 'really', 'um', 'uh', 'er', 'erm', 'let', 'us', 'there']);

const NUMBER_WORDS = { 10: 'ten' };

// English words whose BSL sign is filed under another word.
const SIGN_FOR = { i: 'me', myself: 'me', mine: 'my', yours: 'your', ourselves: 'we', hey: 'hello', hiya: 'hi' };

const isLetter = k => /^[a-z]$/.test(k);  // single letters are only used for fingerspelling

export function tokenize(text) {
  let s = text.toLowerCase().replace(/[’‘]/g, "'");
  s = s.replace(/[a-z]+'[a-z]+/g, m => CONTRACTIONS[m] || m.replace(/'s$/, '').replace(/'/g, ''));
  return s.replace(/[^a-z0-9'\s-]/g, ' ').replace(/'/g, '').split(/[\s-]+/).filter(Boolean)
    .map(t => NUMBER_WORDS[t] || t);
}

function stems(w) {
  const out = [];
  if (w.endsWith('ies') && w.length > 4) out.push(w.slice(0, -3) + 'y');
  if (w.endsWith('es') && w.length > 3) out.push(w.slice(0, -2));
  if (w.endsWith('s') && w.length > 2) out.push(w.slice(0, -1));
  if (w.endsWith('ing') && w.length > 4) {
    const b = w.slice(0, -3);
    out.push(b, b + 'e');
    if (b.length > 2 && b.at(-1) === b.at(-2)) out.push(b.slice(0, -1));
  }
  if (w.endsWith('ed') && w.length > 3) {
    const b = w.slice(0, -2);
    out.push(b, w.slice(0, -1));
    if (b.length > 2 && b.at(-1) === b.at(-2)) out.push(b.slice(0, -1));
    if (w.endsWith('ied')) out.push(w.slice(0, -3) + 'y');
  }
  if (w.endsWith('ly') && w.length > 4) out.push(w.slice(0, -2));
  if (w.endsWith('er') && w.length > 4) out.push(w.slice(0, -2), w.slice(0, -1));
  return out;
}

export function createLookup(clipIndex, synonymIndex) {
  const words = clipIndex.words || {};
  const alternatives = clipIndex.alternatives || {};
  const synonyms = (synonymIndex && synonymIndex.synonyms) || {};
  const irregular = (synonymIndex && synonymIndex.irregular) || {};

  const direct = key => {
    if (isLetter(key)) return null;
    if (words[key]) return key;
    const alt = alternatives[key];
    return alt && words[alt] && !isLetter(alt) ? alt : null;
  };

  function resolveWord(w) {
    if (SIGN_FOR[w] && words[SIGN_FOR[w]]) return { key: SIGN_FOR[w], via: 'word' };
    let hit = direct(w);
    if (hit) return { key: hit, via: hit === w ? 'word' : 'alternative' };
    if (irregular[w] && words[irregular[w]]) return { key: irregular[w], via: 'irregular' };
    for (const s of stems(w)) {
      hit = direct(s) || (irregular[s] && words[irregular[s]] ? irregular[s] : null);
      if (hit) return { key: hit, via: 'root' };
    }
    for (const s of [w, ...stems(w)]) {
      if (synonyms[s] && words[synonyms[s]]) return { key: synonyms[s], via: 'synonym' };
    }
    return null;
  }

  // Letters without a clip still appear (as a text card) so the spelling stays complete.
  function fingerspell(w) {
    return [...w].filter(ch => /[a-z0-9]/.test(ch)).map(ch => ({
      kind: 'letter', label: ch.toUpperCase(), word: w, url: words[ch] ? words[ch].clips[0] : null,
    }));
  }

  // Returns the playlist for a piece of text.
  return function translate(text) {
    const tokens = tokenize(text);
    const items = [];
    for (let i = 0; i < tokens.length;) {
      let matched = false;
      for (let n = Math.min(3, tokens.length - i); n >= 2; n--) {
        const key = direct(tokens.slice(i, i + n).join('-'));
        if (key) {
          items.push({ kind: 'sign', label: words[key].label, heard: tokens.slice(i, i + n).join(' '), via: 'phrase', url: words[key].clips[0] });
          i += n; matched = true; break;
        }
      }
      if (matched) continue;
      const w = tokens[i];
      const hit = resolveWord(w);
      if (hit) {
        items.push({ kind: 'sign', label: words[hit.key].label, heard: w, via: hit.via, url: words[hit.key].clips[0] });
      } else if (!SKIP.has(w)) {
        items.push(...fingerspell(w));
      }
      i++;
    }
    return items;
  };
}
