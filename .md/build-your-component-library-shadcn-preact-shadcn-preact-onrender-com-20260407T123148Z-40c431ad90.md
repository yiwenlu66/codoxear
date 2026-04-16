<!-- TOC START min:2 max:4 -->

* [1 Create project](#1-create-project)
* [2 Add Tailwind and its configuration](#2-add-tailwind-and-its-configuration)
* [3 Edit `tsconfig.json` file](#3-edit-tsconfigjson-file)
* [4 Update `vite.config.ts`](#4-update-viteconfigts)
* [5 Add UI components](#5-add-ui-components)
* [6 Done](#6-done)

<!-- TOC END -->

# Build your component library - shadcn-preact

[shadcn-preact](https://shadcn-preact.onrender.com/)

[Docs](https://shadcn-preact.onrender.com/docs)[Components](https://shadcn-preact.onrender.com/component/accordion)[Examples](https://shadcn-preact.onrender.com/examples)

Search documentation...Search...⌘K

[](https://github.com/LiasCode/shadcn-preact)

[Version 4 preview here](https://v4-shadcn-preact.onrender.com/)

Getting Started[Introduction](https://shadcn-preact.onrender.com/docs)[Installation](https://shadcn-preact.onrender.com/docs/installation)[Theming](https://shadcn-preact.onrender.com/docs/theming)[Dark Mode](https://shadcn-preact.onrender.com/docs/dark)[Examples](https://shadcn-preact.onrender.com/examples)[Tailwind v4](https://v4-shadcn-preact.onrender.com/)

Installation[Vite](https://shadcn-preact.onrender.com/docs/installation/vite)[Astro](https://shadcn-preact.onrender.com/docs/installation/astro)

Components[accordion](https://shadcn-preact.onrender.com/component/accordion)[alert](https://shadcn-preact.onrender.com/component/alert)[alert dialog](https://shadcn-preact.onrender.com/component/alert-dialog)[aspect ratio](https://shadcn-preact.onrender.com/component/aspect-ratio)[avatar](https://shadcn-preact.onrender.com/component/avatar)[badge](https://shadcn-preact.onrender.com/component/badge)[breadcrumb](https://shadcn-preact.onrender.com/component/breadcrumb)[button](https://shadcn-preact.onrender.com/component/button)[calendar](https://shadcn-preact.onrender.com/component/calendar)[card](https://shadcn-preact.onrender.com/component/card)[carousel](https://shadcn-preact.onrender.com/component/carousel)[chart](https://shadcn-preact.onrender.com/component/chart)[checkbox](https://shadcn-preact.onrender.com/component/checkbox)[collapsible](https://shadcn-preact.onrender.com/component/collapsible)[command](https://shadcn-preact.onrender.com/component/command)[dialog](https://shadcn-preact.onrender.com/component/dialog)[drawer](https://shadcn-preact.onrender.com/component/drawer)[input](https://shadcn-preact.onrender.com/component/input)[input otp](https://shadcn-preact.onrender.com/component/input_otp)[label](https://shadcn-preact.onrender.com/component/label)[popover](https://shadcn-preact.onrender.com/component/popover)[progress](https://shadcn-preact.onrender.com/component/progress)[resizable](https://shadcn-preact.onrender.com/component/resizable)[select](https://shadcn-preact.onrender.com/component/select)[separator](https://shadcn-preact.onrender.com/component/separator)[sheet](https://shadcn-preact.onrender.com/component/sheet)[skeleton](https://shadcn-preact.onrender.com/component/skeleton)[switch](https://shadcn-preact.onrender.com/component/switch)[table](https://shadcn-preact.onrender.com/component/table)[tabs](https://shadcn-preact.onrender.com/component/tabs)[textarea](https://shadcn-preact.onrender.com/component/textarea)[toast](https://shadcn-preact.onrender.com/component/toast)[toggle](https://shadcn-preact.onrender.com/component/toggle)[tooltip](https://shadcn-preact.onrender.com/component/tooltip)

1. [Docs](https://shadcn-preact.onrender.com/docs)

2. [Installation](https://shadcn-preact.onrender.com/docs/installation)

3. Vite

vite installation

How to install dependencies and structure your app with vite.

## 1 Create project

Start by creating a new Preact project using create-preact:

##### This guide uses bun as the package manager, but you can also use `npm/pnpm/yarn`.

```
bun create preact@latest
```

## 2 Add Tailwind and its configuration

##### TailwindCSS version

For now only supports TailwindCSS 3. In the future will support TailwindCSS 4.

Install tailwindcss and its peer dependencies.

```
bun add -D tailwindcss@3.4.17 postcss autoprefixer
```

Add this import header in your main css file,`src/index.css` in our case:

```
@tailwind base;
  @tailwind components;
  @tailwind utilities;

  /* ... */
```

Configure template paths in `tailwind.config.js`:

```
/** @type {import('tailwindcss').Config} */
  export default {
    content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
    theme: {
      extend: {},
    },
    plugins: [],
  };
```

Configure the postcss file `postcss.config.js`:

```
export default {
    plugins: {
      tailwindcss: {},
      autoprefixer: {},
    },
  };
```

## 3 Edit `tsconfig.json` file

```
{
    "compilerOptions": {
      "baseUrl": "./",
      "paths": {
        "@/*": ["./src/*"],
        "@ui/*": ["./src/components/ui/*"]
      }
    }
  }
```

## 4 Update `vite.config.ts`

```
bun add -D @types/node
```

```
import { resolve } from "node:path";
  import preact from "@preact/preset-vite";
  import { defineConfig } from "vite";

  // https://vitejs.dev/config/
  export default defineConfig({
    plugins: [preact()],
    server: {
      host: true,
    },
    resolve: {
      alias: {
        "@ui": resolve(resolve(__dirname), "./src/components/ui/"),
        "@": resolve(resolve(__dirname), "./src/"),
      },
    },
    define: {
      "process.env.IS_PREACT": JSON.stringify("true"),
    },
  });
```

## 5 Add UI components

For now this guide its for the installation of all components at once.

Install all components dependencies:

```
bun add class-variance-authority clsx cmdk date-fns dayjs embla-carousel-react input-otp lucide-preact react-day-picker react-hot-toast recharts tailwind-merge tailwindcss-animate vaul @floating-ui/react-dom react-resizable-panels
```

Copy the folder:

Copy the folder of this repo`src/components/ui` into your ui path If you dont change the config guide should be in `src/components/ui`

```
bunx degit https://github.com/LiasCode/shadcn-preact/src/components/ui#v3 ./src/components/ui
```

Adding custom CSS variables:

```
@layer base {
    :root {
      --background: 0 0% 100%;
      --foreground: 240 10% 3.9%;
      --card: 0 0% 100%;
      --card-foreground: 240 10% 3.9%;
      --popover: 0 0% 100%;
      --popover-foreground: 240 10% 3.9%;
      --primary: 240 5.9% 10%;
      --primary-foreground: 0 0% 98%;
      --secondary: 240 4.8% 95.9%;
      --secondary-foreground: 240 5.9% 10%;
      --muted: 240 4.8% 95.9%;
      --muted-foreground: 240 3.8% 46.1%;
      --accent: 240 4.8% 95.9%;
      --accent-foreground: 240 5.9% 10%;
      --destructive: 0 84.2% 60.2%;
      --destructive-foreground: 0 0% 98%;
      --border: 240 5.9% 90%;
      --input: 240 5.9% 90%;
      --ring: 240 5.9% 10%;
      --radius: 0.5rem;
      --chart-1: 12 76% 61%;
      --chart-2: 173 58% 39%;
      --chart-3: 197 37% 24%;
      --chart-4: 43 74% 66%;
      --chart-5: 27 87% 67%;
    }

    .dark {
      --background: 240 10% 3.9%;
      --foreground: 0 0% 98%;
      --card: 240 10% 3.9%;
      --card-foreground: 0 0% 98%;
      --popover: 240 10% 3.9%;
      --popover-foreground: 0 0% 98%;
      --primary: 0 0% 98%;
      --primary-foreground: 240 5.9% 10%;
      --secondary: 240 3.7% 15.9%;
      --secondary-foreground: 0 0% 98%;
      --muted: 240 3.7% 15.9%;
      --muted-foreground: 240 5% 64.9%;
      --accent: 240 3.7% 15.9%;
      --accent-foreground: 0 0% 98%;
      --destructive: 0 62.8% 30.6%;
      --destructive-foreground: 0 0% 98%;
      --border: 240 3.7% 15.9%;
      --input: 240 3.7% 15.9%;
      --ring: 240 4.9% 83.9%;
      --chart-1: 220 70% 50%;
      --chart-2: 160 60% 45%;
      --chart-3: 30 80% 55%;
      --chart-4: 280 65% 60%;
      --chart-5: 340 75% 55%;
    }
  }

  * {
    @apply border-border;
  }

  body {
    @apply bg-background text-foreground;
  }
```

Updating `tailwind.config.js`:

```
/** @type {import('tailwindcss').Config} */
  export default {
    content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
    darkMode: ["class"],
    theme: {
      extend: {
        colors: {
          background: "hsl(var(--background))",
          foreground: "hsl(var(--foreground))",
          card: {
            DEFAULT: "hsl(var(--card))",
            foreground: "hsl(var(--card-foreground))",
          },
          popover: {
            DEFAULT: "hsl(var(--popover))",
            foreground: "hsl(var(--popover-foreground))",
          },
          primary: {
            DEFAULT: "hsl(var(--primary))",
            foreground: "hsl(var(--primary-foreground))",
          },
          secondary: {
            DEFAULT: "hsl(var(--secondary))",
            foreground: "hsl(var(--secondary-foreground))",
          },
          muted: {
            DEFAULT: "hsl(var(--muted))",
            foreground: "hsl(var(--muted-foreground))",
          },
          accent: {
            DEFAULT: "hsl(var(--accent))",
            foreground: "hsl(var(--accent-foreground))",
          },
          destructive: {
            DEFAULT: "hsl(var(--destructive))",
            foreground: "hsl(var(--destructive-foreground))",
          },
          border: "hsl(var(--border))",
          input: "hsl(var(--input))",
          ring: "hsl(var(--ring))",
          chart: {
            1: "hsl(var(--chart-1))",
            2: "hsl(var(--chart-2))",
            3: "hsl(var(--chart-3))",
            4: "hsl(var(--chart-4))",
            5: "hsl(var(--chart-5))",
          },
        },
        fontSize: {},
        borderRadius: {
          "2xl": "calc(var(--radius) + 4px)",
          xl: "calc(var(--radius) + 2px)",
          lg: "var(--radius)",
          md: "calc(var(--radius) - 2px)",
          sm: "calc(var(--radius) - 4px)",
        },
        keyframes: {
          "caret-blink": {
            "0%,70%,100%": { opacity: "1" },
            "20%,50%": { opacity: "0" },
          },
        },
        animation: {
          "caret-blink": "caret-blink 1.25s ease-out infinite",
        },
      },
    },
    plugins: [require("tailwindcss-animate")],
  };
```

## 6 Done

Setup is complete, and your environment is ready.

* [Installation](https://shadcn-preact.onrender.com/docs/installation)
* [Theming](https://shadcn-preact.onrender.com/docs/theming)

Built & designed by [shadcn](https://twitter.com/shadcn). Ported to Preact by [LiasCode](https://github.com/LiasCode). The source code is available on [GitHub.](https://github.com/LiasCode/shadcn-preact)
