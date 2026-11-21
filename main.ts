// Sokoban solver for make.fun sokoban-solver-01-run-20261121 (@arifbrp entry).
//
// Strategy (written fresh in-session, push-based, room by room):
//   Every generated level is one hallway row with 1-4 independent rooms off it
//   (each room 3-4 boxes, boxes can never leave their room in an optimal
//   solution, and the push-optimal count P* is the sum of the rooms' optima).
//   So the level is decomposed into rooms, each room is solved
//   PUSH-OPTIMALLY with iterative-deepening A* (IDA*) over pushes, and the
//   rooms are stitched together with BFS player walks through the hallway.
//   * A push state is (sorted box cells, canonical player region): after a
//     push the player is normalised to the minimum cell of its reachable
//     region, so walking moves never enter the search.
//   * Heuristic: minimum-cost assignment of boxes to goals (bitmask DP over
//     <= 4 boxes) on BFS distances through the room. Admissible: one push
//     moves one box by one cell, so the assignment sum can drop by at most 1.
//   * Deadlock pruning: dead squares (room cells from which no goal is
//     reachable by a box at all) plus a frozen-box test (a box off-goal with
//     all four neighbours wall-or-box can never move again).
//   * Fallbacks, in order: greedy best-first push search per room, then a
//     whole-level push IDA* (same engine, one big "room"). A room that still
//     fails leaves its instance empty rather than emitting an illegal move.
//   * Deterministic: no clock reads, no randomness; every search is bounded
//     by a shared node budget, so two runs write identical bytes.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const DIRS = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

// Deterministic work budgets (nodes expanded), shared across the whole run
// so the total work - and therefore the output - never depends on timing.
const IDA_CAP = 150_000;
const GREEDY_CAP = 120_000;
const GLOBAL_CAP = 400_000;
const budget = { left: 1_500_000 };

interface Level {
  w: number;
  h: number;
  wall: boolean[];
  goals: number[];
  goalSet: Set<number>;
  boxes: number[];
  player: number;
}

interface PushStep {
  box: number;
  dir: number; // index into DIRS
}

interface RoomCtx {
  lvl: Level;
  roomCells: Set<number>;
  goals: number[]; // sorted
  dead: boolean[]; // per cell: no goal reachable by a box from here
  dist: number[][]; // dist[g][cell]: BFS distance through roomCells from goals[g]
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall: boolean[] = new Array(w * h).fill(true);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#';
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet: new Set(goals), boxes, player };
}

function cellX(c: number, w: number): number {
  return c % w;
}
function cellY(c: number, w: number): number {
  return Math.floor(c / w);
}

// Shortest player walk from `from` to `to`, avoiding walls and `blocked`
// cells; null when unreachable. Fixed direction order => deterministic.
function bfsPath(from: number, to: number, lvl: Level, blocked: Set<number>): string | null {
  if (from === to) return '';
  const prev = new Map<number, { p: number; ch: string }>();
  const q: number[] = [from];
  prev.set(from, { p: -1, ch: '' });
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    for (const d of DIRS) {
      const nx = cellX(c, lvl.w) + d.dx;
      const ny = cellY(c, lvl.w) + d.dy;
      if (nx < 0 || nx >= lvl.w || ny < 0 || ny >= lvl.h) continue;
      const n = ny * lvl.w + nx;
      if (lvl.wall[n] || blocked.has(n) || prev.has(n)) continue;
      prev.set(n, { p: c, ch: d.ch });
      if (n === to) {
        let s = '';
        let cur = n;
        while (cur !== from) {
          const e = prev.get(cur)!;
          s = e.ch + s;
          cur = e.p;
        }
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

// All cells the player can reach from `start` without crossing walls or
// `blocked` cells.
function regionSet(start: number, lvl: Level, blocked: Set<number>): Set<number> {
  const seen = new Set<number>([start]);
  const q: number[] = [start];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    for (const d of DIRS) {
      const nx = cellX(c, lvl.w) + d.dx;
      const ny = cellY(c, lvl.w) + d.dy;
      if (nx < 0 || nx >= lvl.w || ny < 0 || ny >= lvl.h) continue;
      const n = ny * lvl.w + nx;
      if (lvl.wall[n] || blocked.has(n) || seen.has(n)) continue;
      seen.add(n);
      q.push(n);
    }
  }
  return seen;
}

// BFS distances from `start` through `allowed` cells only (-1 = unreachable).
function bfsDistances(start: number, lvl: Level, allowed: Set<number>): number[] {
  const dist: number[] = new Array(lvl.w * lvl.h).fill(-1);
  dist[start] = 0;
  const q: number[] = [start];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    for (const d of DIRS) {
      const nx = cellX(c, lvl.w) + d.dx;
      const ny = cellY(c, lvl.w) + d.dy;
      if (nx < 0 || nx >= lvl.w || ny < 0 || ny >= lvl.h) continue;
      const n = ny * lvl.w + nx;
      if (dist[n] >= 0 || !allowed.has(n)) continue;
      dist[n] = dist[c]! + 1;
      q.push(n);
    }
  }
  return dist;
}

function makeCtx(lvl: Level, roomCells: Set<number>, goals: number[]): RoomCtx {
  const dist = goals.map((g) => bfsDistances(g, lvl, roomCells));
  const dead: boolean[] = new Array(lvl.w * lvl.h).fill(false);
  for (const c of roomCells) {
    if (lvl.goalSet.has(c)) continue;
    dead[c] = dist.every((d) => d[c]! < 0);
  }
  return { lvl, roomCells, goals, dead, dist };
}

function popcount(n: number): number {
  let c = 0;
  while (n > 0) {
    c += n & 1;
    n >>= 1;
  }
  return c;
}

// Minimum-cost assignment of boxes to goals on room distances.
// Infinity when some box cannot reach any unassigned goal.
function heuristic(ctx: RoomCtx, boxes: number[]): number {
  const n = boxes.length;
  const INF = 1e9;
  const dp: number[] = new Array(1 << n).fill(INF);
  dp[0] = 0;
  for (let mask = 0; mask < (1 << n); mask++) {
    const k = popcount(mask);
    if (k >= n || dp[mask]! >= INF) continue;
    for (let g = 0; g < n; g++) {
      if (mask & (1 << g)) continue;
      const d = ctx.dist[g]![boxes[k]!]!;
      if (d < 0) continue;
      const nm = mask | (1 << g);
      const v = dp[mask]! + d;
      if (v < dp[nm]!) dp[nm] = v;
    }
  }
  const r = dp[(1 << n) - 1]!;
  return r >= INF ? Infinity : r;
}

// A box off-goal whose four neighbours are all wall-or-box can never move
// again (only pushes move boxes, and every push direction is blocked).
function anyFrozen(ctx: RoomCtx, boxes: number[]): boolean {
  const lvl = ctx.lvl;
  const s = new Set(boxes);
  for (const b of boxes) {
    if (lvl.goalSet.has(b)) continue;
    let stuck = true;
    for (const d of DIRS) {
      const nx = cellX(b, lvl.w) + d.dx;
      const ny = cellY(b, lvl.w) + d.dy;
      if (nx < 0 || nx >= lvl.w || ny < 0 || ny >= lvl.h) continue;
      const n = ny * lvl.w + nx;
      if (!lvl.wall[n] && !s.has(n)) {
        stuck = false;
        break;
      }
    }
    if (stuck) return true;
  }
  return false;
}

function solved(ctx: RoomCtx, boxes: number[]): boolean {
  return boxes.every((b) => ctx.lvl.goalSet.has(b));
}

// All legal pushes from a state, each with its child heuristic, best first.
function successors(
  ctx: RoomCtx,
  boxes: number[],
  player: number,
  blocked: Set<number>,
): { step: PushStep; child: number[]; h: number }[] {
  const lvl = ctx.lvl;
  const boxSet = new Set(boxes);
  const blockedAll = new Set(blocked);
  for (const b of boxes) blockedAll.add(b);
  const region = regionSet(player, lvl, blockedAll);
  const out: { step: PushStep; child: number[]; h: number }[] = [];
  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i]!;
    for (let di = 0; di < 4; di++) {
      const d = DIRS[di]!;
      const tx = cellX(b, lvl.w) + d.dx;
      const ty = cellY(b, lvl.w) + d.dy;
      const sx = cellX(b, lvl.w) - d.dx;
      const sy = cellY(b, lvl.w) - d.dy;
      if (tx < 0 || tx >= lvl.w || ty < 0 || ty >= lvl.h) continue;
      if (sx < 0 || sx >= lvl.w || sy < 0 || sy >= lvl.h) continue;
      const target = ty * lvl.w + tx;
      const stance = sy * lvl.w + sx;
      if (!ctx.roomCells.has(target)) continue;
      if (boxSet.has(target)) continue;
      if (ctx.dead[target]) continue;
      if (!region.has(stance)) continue;
      const child = boxes.slice();
      child[i] = target;
      child.sort((a, b2) => a - b2);
      if (anyFrozen(ctx, child)) continue;
      const h = heuristic(ctx, child);
      if (!isFinite(h)) continue;
      out.push({ step: { box: b, dir: di }, child, h });
    }
  }
  out.sort((a, b) => a.h - b.h || a.step.box - b.step.box || a.step.dir - b.step.dir);
  return out;
}

interface DfsResult {
  found: boolean;
  next: number;
  path?: PushStep[];
}

// IDA* over pushes. `tt` is per-iteration: a state revisited in the same
// iteration with g' <= g cannot yield a new solution under this bound.
function dfs(
  ctx: RoomCtx,
  boxes: number[],
  player: number,
  g: number,
  bound: number,
  path: PushStep[],
  tt: Map<string, number>,
  blocked: Set<number>,
  st: { aborted: boolean },
): DfsResult {
  const h = heuristic(ctx, boxes);
  if (!isFinite(h)) return { found: false, next: Infinity };
  const f = g + h;
  if (f > bound) return { found: false, next: f };
  if (solved(ctx, boxes)) return { found: true, next: bound, path: path.slice() };
  const blockedAll = new Set(blocked);
  for (const b of boxes) blockedAll.add(b);
  const region = regionSet(player, ctx.lvl, blockedAll);
  let min = Infinity;
  for (const c of region) if (c < min) min = c;
  const key = boxes.join(',') + ':' + min;
  const seen = tt.get(key);
  if (seen !== undefined && seen <= g) return { found: false, next: Infinity };
  tt.set(key, g);
  if (budget.left-- <= 0) {
    st.aborted = true;
    return { found: false, next: Infinity };
  }
  let next = Infinity;
  for (const s of successors(ctx, boxes, player, blocked)) {
    path.push(s.step);
    const r = dfs(ctx, s.child, s.step.box, g + 1, bound, path, tt, blocked, st);
    path.pop();
    if (r.found) return r;
    if (st.aborted) return { found: false, next: Infinity };
    if (r.next < next) next = r.next;
  }
  return { found: false, next };
}

function idaStar(
  ctx: RoomCtx,
  startBoxes: number[],
  entry: number,
  blocked: Set<number>,
  cap: number,
): PushStep[] | null {
  const h0 = heuristic(ctx, startBoxes);
  if (!isFinite(h0)) return null;
  if (solved(ctx, startBoxes)) return [];
  let bound = h0;
  const startBudget = budget.left;
  const st = { aborted: false };
  for (;;) {
    const tt = new Map<string, number>();
    const path: PushStep[] = [];
    const r = dfs(ctx, startBoxes, entry, 0, bound, path, tt, blocked, st);
    if (r.found) return r.path ?? [];
    if (st.aborted || !isFinite(r.next)) return null;
    if (startBudget - budget.left > cap) return null;
    bound = r.next;
  }
}

// Greedy best-first push search (first solution found, not necessarily
// optimal). Used when IDA* exhausts its node cap.
function greedyPushes(
  ctx: RoomCtx,
  startBoxes: number[],
  entry: number,
  blocked: Set<number>,
  cap: number,
): PushStep[] | null {
  const startBudget = budget.left;
  const keyOf = (boxes: number[], player: number): string => {
    const blockedAll = new Set(blocked);
    for (const b of boxes) blockedAll.add(b);
    const region = regionSet(player, ctx.lvl, blockedAll);
    let min = Infinity;
    for (const c of region) if (c < min) min = c;
    return boxes.join(',') + ':' + min;
  };
  const boxesArr: number[][] = [startBoxes.slice()];
  const playerArr: number[] = [entry];
  const gArr: number[] = [0];
  const parentArr: number[] = [-1];
  const stepArr: (PushStep | null)[] = [null];
  // Binary heap on (h, g, idx): lowest h first.
  const heap: number[] = [];
  const hArr: number[] = [heuristic(ctx, startBoxes)];
  const less = (a: number, b: number): boolean =>
    hArr[a]! < hArr[b]! || (hArr[a]! === hArr[b]! && (gArr[a]! < gArr[b]! || (gArr[a]! === gArr[b]! && a < b)));
  const heapPush = (idx: number): void => {
    let i = heap.length;
    heap.push(idx);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const t = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const t = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };
  heapPush(0);
  const best = new Map<string, number>([[keyOf(startBoxes, entry), 0]]);
  while (heap.length > 0) {
    if (budget.left-- <= 0 || startBudget - budget.left > cap) return null;
    const s = heapPop();
    const boxes = boxesArr[s]!;
    if (solved(ctx, boxes)) {
      const path: PushStep[] = [];
      for (let t = s; t > 0; t = parentArr[t]!) path.unshift(stepArr[t]!);
      return path;
    }
    for (const succ of successors(ctx, boxes, playerArr[s]!, blocked)) {
      const k = keyOf(succ.child, succ.step.box);
      const seen = best.get(k);
      if (seen !== undefined && seen <= gArr[s]! + 1) continue;
      best.set(k, gArr[s]! + 1);
      const t = boxesArr.length;
      boxesArr.push(succ.child);
      playerArr.push(succ.step.box);
      gArr.push(gArr[s]! + 1);
      hArr.push(succ.h);
      parentArr.push(s);
      stepArr.push(succ.step);
      heapPush(t);
    }
  }
  return null;
}

// Turn a push sequence into player moves, walking between pushes.
function pushesToMoves(
  ctx: RoomCtx,
  startBoxes: number[],
  entry: number,
  blocked: Set<number>,
  pushes: PushStep[],
): { moves: string; endPlayer: number } | null {
  const lvl = ctx.lvl;
  let boxes = startBoxes.slice();
  let player = entry;
  let moves = '';
  for (const p of pushes) {
    const d = DIRS[p.dir]!;
    const stance = p.box - (d.dy * lvl.w + d.dx);
    const blockedNow = new Set(blocked);
    for (const b of boxes) blockedNow.add(b);
    const walk = bfsPath(player, stance, lvl, blockedNow);
    if (walk === null) return null;
    moves += walk;
    moves += d.ch;
    const idx = boxes.indexOf(p.box);
    if (idx < 0) return null;
    boxes[idx] = p.box + (d.dy * lvl.w + d.dx);
    boxes.sort((a, b) => a - b);
    player = p.box;
  }
  return { moves, endPlayer: player };
}

function solveRoom(
  lvl: Level,
  roomCells: Set<number>,
  roomGoals: number[],
  roomBoxes: number[],
  entry: number,
  blocked: Set<number>,
  cap: number,
): { moves: string; endPlayer: number; pushes: number } | null {
  const ctx = makeCtx(lvl, roomCells, roomGoals);
  let pushes = idaStar(ctx, roomBoxes, entry, blocked, cap);
  if (!pushes) pushes = greedyPushes(ctx, roomBoxes, entry, blocked, GREEDY_CAP);
  if (!pushes) return null;
  const conv = pushesToMoves(ctx, roomBoxes, entry, blocked, pushes);
  if (!conv) return null;
  return { moves: conv.moves, endPlayer: conv.endPlayer, pushes: pushes.length };
}

interface Room {
  cells: Set<number>;
  boxes: number[];
  goals: number[];
}

// Split the level into rooms: connected components of floor cells outside
// the player's (hallway) row. Boxes/goals sitting in the hallway row belong
// to the component touching them. Returns null when the split is not clean.
function decompose(lvl: Level): Room[] | null {
  const hy = cellY(lvl.player, lvl.w);
  const inRow = (c: number): boolean => cellY(c, lvl.w) === hy;
  const seen = new Set<number>();
  const comps: Set<number>[] = [];
  for (let c = 0; c < lvl.w * lvl.h; c++) {
    if (lvl.wall[c] || inRow(c) || seen.has(c)) continue;
    const comp = new Set<number>();
    const q: number[] = [c];
    seen.add(c);
    comp.add(c);
    for (let qi = 0; qi < q.length; qi++) {
      const cur = q[qi]!;
      for (const d of DIRS) {
        const nx = cellX(cur, lvl.w) + d.dx;
        const ny = cellY(cur, lvl.w) + d.dy;
        if (nx < 0 || nx >= lvl.w || ny < 0 || ny >= lvl.h) continue;
        const n = ny * lvl.w + nx;
        if (lvl.wall[n] || inRow(n) || seen.has(n)) continue;
        seen.add(n);
        comp.add(n);
        q.push(n);
      }
    }
    comps.push(comp);
  }
  const rooms: Room[] = comps.map((cells) => ({ cells, boxes: [], goals: [] }));
  const adjacentRoom = (c: number): Room | null => {
    for (const r of rooms) {
      for (const d of DIRS) {
        const nx = cellX(c, lvl.w) + d.dx;
        const ny = cellY(c, lvl.w) + d.dy;
        if (nx < 0 || nx >= lvl.w || ny < 0 || ny >= lvl.h) continue;
        if (r.cells.has(ny * lvl.w + nx)) return r;
      }
    }
    return null;
  };
  for (const b of lvl.boxes) {
    if (inRow(b)) {
      const r = adjacentRoom(b);
      if (!r) return null;
      r.boxes.push(b);
    } else {
      const r = rooms.find((rm) => rm.cells.has(b));
      if (!r) return null;
      r.boxes.push(b);
    }
  }
  for (const g of lvl.goals) {
    if (inRow(g)) {
      const r = adjacentRoom(g);
      if (!r) return null;
      r.goals.push(g);
    } else {
      const r = rooms.find((rm) => rm.cells.has(g));
      if (!r) return null;
      r.goals.push(g);
    }
  }
  for (const r of rooms) {
    r.boxes.sort((a, b) => a - b);
    r.goals.sort((a, b) => a - b);
  }
  // Every box/goal must belong to exactly one room, counts must match.
  let assignedBoxes = 0;
  let assignedGoals = 0;
  for (const r of rooms) {
    if (r.boxes.length === 0 || r.boxes.length !== r.goals.length) return null;
    assignedBoxes += r.boxes.length;
    assignedGoals += r.goals.length;
  }
  if (assignedBoxes !== lvl.boxes.length || assignedGoals !== lvl.goals.length) return null;
  rooms.sort((a, b) => Math.min(...a.cells) - Math.min(...b.cells));
  return rooms;
}

function verify(lvl: Level, moves: string): boolean {
  let player = lvl.player;
  const boxes = new Set(lvl.boxes);
  for (const ch of moves) {
    if (ch === '\n' || ch === ' ' || ch === '\r' || ch === '\t') continue;
    const lower = ch.toLowerCase();
    const d = DIRS.find((dd) => dd.ch === lower);
    if (!d) return false;
    const nx = cellX(player, lvl.w) + d.dx;
    const ny = cellY(player, lvl.w) + d.dy;
    if (nx < 0 || nx >= lvl.w || ny < 0 || ny >= lvl.h) return false;
    const q = ny * lvl.w + nx;
    if (lvl.wall[q]) return false;
    if (boxes.has(q)) {
      const tx = nx + d.dx;
      const ty = ny + d.dy;
      if (tx < 0 || tx >= lvl.w || ty < 0 || ty >= lvl.h) return false;
      const t = ty * lvl.w + tx;
      if (lvl.wall[t] || boxes.has(t)) return false;
      boxes.delete(q);
      boxes.add(t);
    }
    player = q;
  }
  for (const b of boxes) if (!lvl.goalSet.has(b)) return false;
  return true;
}

function solveByRooms(lvl: Level, rooms: Room[]): string | null {
  const hy = cellY(lvl.player, lvl.w);
  let boxes = lvl.boxes.slice();
  let player = lvl.player;
  let moves = '';
  for (const room of rooms) {
    const roomBoxSet = new Set(room.boxes);
    const blocked = new Set<number>();
    for (const b of boxes) if (!roomBoxSet.has(b)) blocked.add(b);
    // Entry cells: hallway-row floor cells touching the room.
    const entries = new Set<number>();
    for (const c of room.cells) {
      for (const d of DIRS) {
        const nx = cellX(c, lvl.w) + d.dx;
        const ny = cellY(c, lvl.w) + d.dy;
        if (ny !== hy || nx < 0 || nx >= lvl.w || ny < 0 || ny >= lvl.h) continue;
        const n = ny * lvl.w + nx;
        if (!lvl.wall[n]) entries.add(n);
      }
    }
    if (entries.size === 0) return null;
    // Try entries nearest the player first.
    const boxSet = new Set(boxes);
    const entryList = [...entries].sort((a, b) => {
      const wa = bfsPath(player, a, lvl, boxSet);
      const wb = bfsPath(player, b, lvl, boxSet);
      const da = wa === null ? Infinity : wa.length;
      const db = wb === null ? Infinity : wb.length;
      return da - db || a - b;
    });
    let done: { walk: string; r: { moves: string; endPlayer: number; pushes: number } } | null = null;
    for (const e of entryList) {
      const walk = bfsPath(player, e, lvl, boxSet);
      if (walk === null) continue;
      const r = solveRoom(lvl, room.cells, room.goals, room.boxes, e, blocked, IDA_CAP);
      if (r) {
        done = { walk, r };
        break;
      }
    }
    if (!done) return null;
    moves += done.walk + done.r.moves;
    const rest = boxes.filter((b) => !roomBoxSet.has(b));
    for (const g of room.goals) rest.push(g);
    rest.sort((a, b) => a - b);
    boxes = rest;
    player = done.r.endPlayer;
  }
  return verify(lvl, moves) ? moves : null;
}

function solveLevel(text: string): string {
  const lvl = parseLevel(text);
  const rooms = decompose(lvl);
  if (rooms) {
    const byRooms = solveByRooms(lvl, rooms);
    if (byRooms !== null) return byRooms;
  }
  // Whole-level fallback: same push engine, one big room.
  const allCells = new Set<number>();
  for (let c = 0; c < lvl.w * lvl.h; c++) if (!lvl.wall[c]) allCells.add(c);
  const r = solveRoom(lvl, allCells, lvl.goals.slice(), lvl.boxes.slice(), lvl.player, new Set(), GLOBAL_CAP);
  if (r && verify(lvl, r.moves)) return r.moves;
  return '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
