// ============================================
// FILE: server/utils/lruCache.js
// O(1) LRU Cache — Map insertion-order trick.
//
// Map preserves insertion order. On every get() hit
// we delete + re-insert the entry so it moves to the
// tail. The head is always the least-recently-used
// entry and is evicted first when capacity is full.
//
// Optional TTL: entries are treated as missing once
// they exceed their time-to-live, even if still in map.
// ============================================

class LRUCache {
  /**
   * @param {number} capacity  Max entries before LRU eviction
   * @param {number} ttlMs     Max age in ms (0 = no expiry)
   */
  constructor(capacity, ttlMs = 0) {
    this.capacity = capacity;
    this.ttlMs    = ttlMs;
    this._map     = new Map();
    this.hits     = 0;
    this.misses   = 0;
    this.evictions = 0;
  }

  /** @returns {*} value or null (miss / expired) */
  get(key) {
    if (!this._map.has(key)) { this.misses++; return null; }

    const entry = this._map.get(key);

    if (this.ttlMs > 0 && Date.now() - entry.ts > this.ttlMs) {
      this._map.delete(key);
      this.misses++;
      return null;
    }

    // Promote to tail (most-recently-used)
    this._map.delete(key);
    this._map.set(key, entry);
    this.hits++;
    return entry.value;
  }

  /** Store a value; evicts LRU if over capacity */
  set(key, value) {
    if (this._map.has(key)) this._map.delete(key);          // re-insert at tail
    else if (this._map.size >= this.capacity) this._evict(); // make room

    this._map.set(key, { value, ts: Date.now() });
  }

  has(key) {
    return this.get(key) !== null; // respects TTL
  }

  get size()     { return this._map.size; }
  get hitRate()  {
    const total = this.hits + this.misses;
    return total ? ((this.hits / total) * 100).toFixed(1) + "%" : "0%";
  }

  stats() {
    return {
      size: this._map.size,
      capacity: this.capacity,
      ttlMs: this.ttlMs,
      hits: this.hits,
      misses: this.misses,
      evictions: this.evictions,
      hitRate: this.hitRate,
    };
  }

  _evict() {
    const lruKey = this._map.keys().next().value; // head = LRU
    this._map.delete(lruKey);
    this.evictions++;
    console.log(`[LRU] Evicted "${lruKey}" (capacity=${this.capacity})`);
  }
}

module.exports = LRUCache;
