export const coupleName = "Mel & Sam";
export const weddingLabel = `${coupleName} Wedding`;

export const venue = {
	name: "Fernbank Farm",
	lat: -33.2641853,
	lon: 151.3361165,
};

export const googlePlaceUrl =
	"https://www.google.com/maps/place/Fernbank+Farm/@-33.2645291,151.3337427,1175m/data=!3m1!1e3!4m6!3m5!1s0x6b72cd563c66b32b:0x6c1ed69c8423f31a!8m2!3d-33.2641853!4d151.3361165!16s%2Fg%2F11b7c4wx9g?entry=ttu";

export const googleDirectionsUrl = `https://www.google.com/maps/dir/?api=1&destination=${venue.lat},${venue.lon}`;

const bboxDelta = {
	lat: 0.01,
	lon: 0.012,
};

const osmBbox = encodeURIComponent(
	`${venue.lon - bboxDelta.lon},${venue.lat - bboxDelta.lat},${venue.lon + bboxDelta.lon},${venue.lat + bboxDelta.lat}`,
);

const osmMarker = encodeURIComponent(`${venue.lat},${venue.lon}`);

export const osmEmbedUrl = `https://www.openstreetmap.org/export/embed.html?bbox=${osmBbox}&layer=mapnik&marker=${osmMarker}`;

export type ScheduleItem = {
	time: string;
	title: string;
	note?: string;
};

export const schedule: ScheduleItem[] = [
	{ time: "3:00 PM", title: "Guests arrive", note: "Grab a drink, find your seat." },
	{ time: "3:30 PM", title: "Ceremony" },
	{ time: "4:00 PM", title: "Cocktail hour" },
	{ time: "5:30 PM", title: "Dinner" },
	{ time: "7:00 PM", title: "Speeches + cake" },
	{ time: "7:30 PM", title: "Dancing" },
	{ time: "10:30 PM", title: "Farewell" },
];

export type FaqItem = {
	q: string;
	a: string;
};

export const faqs: FaqItem[] = [
	{ q: "What’s the dress code?", a: "TBD (we’ll update this soon)." },
	{ q: "Can I bring a plus-one?", a: "TBD (we’ll update this soon)." },
	{ q: "Are kids welcome?", a: "TBD (we’ll update this soon)." },
	{ q: "Food allergies / dietary needs?", a: "TBD (we’ll update this soon)." },
	{ q: "What about gifts?", a: "TBD (registry coming soon)." },
];
