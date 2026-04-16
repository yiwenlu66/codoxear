<!-- TOC START min:2 max:4 -->

* [Usage](#usage)

* [Examples](#examples)

  * [Custom close button](#custom-close-button)

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

2. dialog

dialog

A window overlaid on either the primary window or another dialog window, rendering the content underneath inert.

Preview Code

Edit Profile

## Usage

```
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
  } from "@ui/dialog"
```

```
<Dialog>
    <DialogTrigger>Open</DialogTrigger>
    <DialogContent>
      <DialogHeader>
        <DialogTitle>Are you absolutely sure?</DialogTitle>
        <DialogDescription>
          This action cannot be undone. This will permanently delete your account
          and remove your data from our servers.
        </DialogDescription>
      </DialogHeader>
    </DialogContent>
  </Dialog>
```

## Examples

### Custom close button

Preview Code

Share

* [Command](https://shadcn-preact.onrender.com/component/command)
* [Drawer](https://shadcn-preact.onrender.com/component/drawer)

Built & designed by [shadcn](https://twitter.com/shadcn). Ported to Preact by [LiasCode](https://github.com/LiasCode). The source code is available on [GitHub.](https://github.com/LiasCode/shadcn-preact)
