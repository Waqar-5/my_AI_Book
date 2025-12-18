import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  // title: 'My Site',
  // tagline: 'Dinosaurs are cool',
  // favicon: 'img/favicon.ico',
title: 'Embodied Intelligence: The Physical AI & Humanoid Robotics Handbook',
  // tagline: 'Building Intelligent Humanoids with ROS 2, Gazebo, Isaac & VLA',
  tagline: 'Learn to Build Intelligent Humanoids: Master ROS 2, Simulate with Gazebo & Unity, Train AI Perception in NVIDIA Isaac, and Enable Robots to Understand Language and Act Autonomously with Vision-Language-Action (VLA).”',
  favicon: 'img/docusaurus.png',
  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://your-docusaurus-site.example.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'Waqar-5', // Usually your GitHub org/user name.
  projectName: 'docBook', // Usually your repo name.

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // editUrl:
          //   'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // editUrl:
          //   'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
        title: 'The Physical AI & Humanoid Robotics Handbook',
      logo: {
  alt: 'Physical AI Book Logo',
  src: '/img/docusaurus.png', // ✅ correct if file is in static/img/
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
            label: 'Physical AI Handbook',
        },
        // {to: '/blog', label: 'Blog', position: 'left'},
         {
          href: 'https://github.com/Waqar-5',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Tutorial',
              to: '/docs/intro',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            // {
            //   label: 'Stack Overflow',
            //   href: 'https://stackoverflow.com/questions/tagged/docusaurus',
            // },
            {
              label: 'LinkedIn',
              href: 'https://www.linkedin.com/in/waqar-ali-997b962b5/',
            },
            {
              label: 'X(Twitter)',
              href: 'https://x.com/docusaurus',
            },
          ],
        },
        {
          title: 'More',
          items: [
            // {
            //   label: 'Blog',
            //   to: '/blog',
            // },
            {
               label: 'GitHub',
              href: 'https://github.com/Waqar-5',
            },
          ],
        },
      ],
       // copyright: `Copyright © ${new Date().getFullYear()} Panaverse DAO. Built with Docusaurus.`,
      copyright: `Copyright © ${new Date().getFullYear()} "The Physical AI & Humanoid Robotics Handbook". 
  Built with ❤️ by <a href="https://github.com/Waqar-5" target="_blank">Waqar Ali</a> — Creator, Educator, and AI Enthusiast.`
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
