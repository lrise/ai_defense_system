# Frontend Dockerfile - 多階段構建
# 第一階段：構建階段
FROM node:18-alpine AS build

WORKDIR /app

# 複製 package.json 和 package-lock.json（如果存在）
COPY package*.json ./

# 安裝依賴
RUN npm ci --only=production

# 複製源代碼
COPY . .

# 構建生產版本
RUN npm run build

# 第二階段：生產階段
FROM nginx:alpine

# 複製自定義 nginx 配置
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 複製構建好的應用
COPY --from=build /app/build /usr/share/nginx/html

# 暴露端口
EXPOSE 80

# 健康檢查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

# 啟動 nginx
CMD ["nginx", "-g", "daemon off;"]