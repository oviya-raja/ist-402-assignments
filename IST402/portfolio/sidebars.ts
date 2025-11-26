import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Creating a sidebar enables you to:
 * - create an ordered group of docs
 * - render a sidebar for each doc of that group
 * - provide next/previous navigation
 *
 * The sidebars can be generated from the filesystem, or explicitly defined here.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    'introduction',
    'course-guide',
    'bloom-taxonomy',
    'documentation-guide',
    {
      type: 'category',
      label: 'Week 3: Prompt Engineering & QA',
      items: [
        'week3-prompt-engineering/overview',
        'week3-prompt-engineering/remember',
        'week3-prompt-engineering/understand',
        'week3-prompt-engineering/apply',
        'week3-prompt-engineering/analyze',
        'week3-prompt-engineering/evaluate',
        'week3-prompt-engineering/create',
      ],
    },
    {
      type: 'category',
      label: 'Week 6: AI Agents with n8n',
      items: [
        'week6-ai-agents-n8n/overview',
        'week6-ai-agents-n8n/remember',
        'week6-ai-agents-n8n/understand',
        'week6-ai-agents-n8n/apply',
        'week6-ai-agents-n8n/analyze',
        'week6-ai-agents-n8n/evaluate',
        'week6-ai-agents-n8n/create',
      ],
    },
    {
      type: 'category',
      label: 'Week 7: Group Assignment',
      items: [
        'week7-group-assignment/overview',
        'week7-group-assignment/remember',
        'week7-group-assignment/understand',
        'week7-group-assignment/apply',
        'week7-group-assignment/analyze',
        'week7-group-assignment/evaluate',
        'week7-group-assignment/create',
      ],
    },
    {
      type: 'category',
      label: 'Week 8: Multimodal AI',
      items: [
        'week8-multimodal/overview',
        'week8-multimodal/remember',
        'week8-multimodal/understand',
        'week8-multimodal/apply',
        'week8-multimodal/analyze',
        'week8-multimodal/evaluate',
        'week8-multimodal/create',
      ],
    },
    {
      type: 'category',
      label: 'Week 9: LlamaIndex',
      items: [
        'week9-llamaindex/overview',
        'week9-llamaindex/remember',
        'week9-llamaindex/understand',
        'week9-llamaindex/apply',
        'week9-llamaindex/analyze',
        'week9-llamaindex/evaluate',
        'week9-llamaindex/create',
      ],
    },
    {
      type: 'category',
      label: 'Week 10: Advanced RAG',
      items: [
        'week10-advanced-rag/overview',
        'week10-advanced-rag/remember',
        'week10-advanced-rag/understand',
        'week10-advanced-rag/apply',
        'week10-advanced-rag/analyze',
        'week10-advanced-rag/evaluate',
        'week10-advanced-rag/create',
      ],
    },
    {
      type: 'category',
      label: 'Week 11: Advanced Topics',
      items: [
        'week11-advanced-topics/overview',
        'week11-advanced-topics/remember',
        'week11-advanced-topics/understand',
        'week11-advanced-topics/apply',
        'week11-advanced-topics/analyze',
        'week11-advanced-topics/evaluate',
        'week11-advanced-topics/create',
      ],
    },
  ],
};

export default sidebars;
