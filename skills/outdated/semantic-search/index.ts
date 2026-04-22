/*
Reference: legacy/python/src/fs_explorer/agent.py
Reference: legacy/python/src/fs_explorer/search/query.py
*/

import type { SkillModule } from "../../../src/types/skills.js";

export const skillModule: SkillModule = {
  tools: {
    async semantic_search() {
      return {
        output: "semantic_search is archived under outdated/ and is not part of the active runtime.",
      };
    },
  },
};

