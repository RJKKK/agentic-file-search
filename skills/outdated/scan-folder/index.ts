/*
Reference: legacy/python/src/fs_explorer/fs.py
Reference: legacy/python/src/fs_explorer/agent.py
*/

import type { SkillModule } from "../../../src/types/skills.js";

export const skillModule: SkillModule = {
  tools: {
    async scan_folder() {
      return {
        output: "scan_folder is archived under outdated/ and is not part of the active runtime.",
      };
    },
  },
};

