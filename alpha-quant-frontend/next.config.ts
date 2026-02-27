// next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 告诉 Next.js 导出纯静态 HTML 文件
  output: 'export',

  // 如果你在代码里用了 <Image /> 标签，还需要加上下面这行，否则打包静态文件时会报错
  images: {
    unoptimized: true,
  },
};

export default nextConfig;