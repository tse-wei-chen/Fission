# syntax=docker/dockerfile:1.7

ARG DOTNET_VERSION=10.0

FROM mcr.microsoft.com/dotnet/sdk:${DOTNET_VERSION} AS build
WORKDIR /src

COPY . .
RUN dotnet restore src/Fission.Server/Fission.Server.csproj
RUN dotnet publish src/Fission.Server/Fission.Server.csproj \
    --configuration Release \
    --no-restore \
    --output /out \
    /p:UseAppHost=false

FROM mcr.microsoft.com/dotnet/aspnet:${DOTNET_VERSION} AS runtime
WORKDIR /app

LABEL org.opencontainers.image.source="https://github.com/tse-wei-chen/Fission" \
      org.opencontainers.image.title="Fission" \
      org.opencontainers.image.description="Experimental high-concurrency LLM inference runtime for .NET"

ENV ASPNETCORE_URLS=http://+:8000 \
    DOTNET_EnableDiagnostics=0 \
    Fission__Device=cpu:0

EXPOSE 8000
STOPSIGNAL SIGINT

COPY --from=build /out ./

USER $APP_UID
ENTRYPOINT ["dotnet", "Fission.Server.dll"]
