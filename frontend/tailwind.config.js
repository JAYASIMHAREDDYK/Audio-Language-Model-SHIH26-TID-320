/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0A0A0C",
        panel: {
          DEFAULT: "#111114",
          raised: "#17171B",
          hover: "#1E1E23",
          active: "#24242B"
        },
        border: {
          hairline: "#242429",
          subtle: "#323238",
          focus: "#E0A458"
        },
        amber: {
          DEFAULT: "#E0A458",
          hover: "#F2BA74",
          subtle: "rgba(224, 164, 88, 0.12)"
        },
        status: {
          critical: "#E5484D",
          warning: "#E0C458",
          nominal: "#3DD68C"
        }
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "SF Mono", "Consolas", "monospace"]
      }
    },
  },
  plugins: [],
}
