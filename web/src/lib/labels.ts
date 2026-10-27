import type { Severity } from "@/api/types";

/** What each RDD2022 class code means, in the long name and in plain words. */
const DAMAGE: Record<string, { name: string; plain: string }> = {
  D00: { name: "longitudinal crack", plain: "a crack that runs along the road" },
  D10: { name: "transverse crack", plain: "a crack that runs across the road" },
  D20: { name: "alligator crack", plain: "a web of cracks, like alligator skin" },
  D40: { name: "pothole", plain: "a pothole" },
};

export function damageName(code: string): string {
  return DAMAGE[code]?.name ?? code;
}

export function damagePlain(code: string): string {
  return DAMAGE[code]?.plain ?? `a defect (${code})`;
}

const OBJECTS: Record<string, { one: string; many: string }> = {
  pedestrian: { one: "person", many: "people" },
  cyclist: { one: "cyclist", many: "cyclists" },
  motorcycle: { one: "motorcycle", many: "motorcycles" },
  car: { one: "car", many: "cars" },
  truck: { one: "truck", many: "trucks" },
  bus: { one: "bus", many: "buses" },
  barrier: { one: "barrier", many: "barriers" },
  traffic_cone: { one: "traffic cone", many: "traffic cones" },
};

function spaced(label: string): string {
  return label.replace(/_/g, " ");
}

export function objectName(label: string, count = 1): string {
  const entry = OBJECTS[label];
  if (entry) return count === 1 ? entry.one : entry.many;
  return count === 1 ? spaced(label) : `${spaced(label)}s`;
}

/** How serious a kind of damage is, in one plain sentence. A guess from the type, not a measurement. */
export const SEVERITY_PLAIN: Record<Severity, string> = {
  low: "On its own this kind of damage is a small worry.",
  medium: "This kind of damage is worth watching.",
  high: "This kind of damage can hurt a tire or a wheel.",
  unknown: "I can't tell how bad this kind of damage is.",
};

/** Colors for the boxes, by severity. Each also has a text label, so color is never the only cue. */
export const SEVERITY_COLOR: Record<Severity, string> = {
  low: "#f2d544",
  medium: "#f39a2d",
  high: "#e5483d",
  unknown: "#8aa0b4",
};
