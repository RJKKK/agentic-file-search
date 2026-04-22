/*
Reference: legacy/python/src/fs_explorer/server.py
Reference: legacy/python/src/fs_explorer/model_config.py
*/

import { loadEnvFile } from "../runtime/env.js";
import { runServer } from "./http-server.js";

loadEnvFile();

const host = process.env.FS_EXPLORER_HOST?.trim() || "127.0.0.1";
const port = Number.parseInt(process.env.FS_EXPLORER_PORT || "8000", 10);

const app = await runServer({
  host,
  port,
  options: {
    logger: Boolean(process.env.FS_EXPLORER_LOG_LEVEL),
  },
});

const shutdown = async (signal: string): Promise<void> => {
  app.log.info({ signal }, "Shutting down FsExplorer Node server");
  await app.close();
  process.exit(0);
};

process.on("SIGINT", () => {
  void shutdown("SIGINT");
});
process.on("SIGTERM", () => {
  void shutdown("SIGTERM");
});
