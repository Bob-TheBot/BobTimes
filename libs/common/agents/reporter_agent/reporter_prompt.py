"""Reporter agent prompt building and formatting."""

from agents.models.story_models import AgentResponse, ResearchResult, StoryDraft, TopicList
from agents.models.task_models import ReporterTask
from agents.reporter_agent.reporter_state import ReporterState, TaskPhase
from agents.reporter_agent.reporter_tools import ReporterToolRegistry
from agents.types import EconomicsSubSection, ReporterField, ScienceSubSection, TaskType, TechnologySubSection
from pydantic import BaseModel

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================


class SubSectionConfig(BaseModel):
    """Configuration for a specific sub-section."""

    guidance: str
    search_terms: str


class FieldConfig(BaseModel):
    """Configuration for a reporter field."""

    guidelines: str
    sub_sections: dict[str, SubSectionConfig]


class ReporterConfiguration:
    """Centralized configuration for reporter agent fields and sub-sections."""

    # Field-specific guidelines
    FIELD_GUIDELINES: dict[ReporterField, str] = {
        ReporterField.TECHNOLOGY: (
            "Focus on innovation, breakthroughs, and industry trends. "
            "Explain technical concepts in accessible language. "
            "Include perspectives from developers, researchers, and users. "
            "Cover both benefits and potential risks of new technologies. "
            "Reference relevant companies, startups, and research institutions."
        ),
        ReporterField.ECONOMICS: (
            "Analyze market trends, policy changes, and economic indicators. "
            "Include data and statistics to support your reporting. "
            "Quote economists, analysts, and industry experts. "
            "Explain complex economic concepts clearly. "
            "Consider both local and global economic impacts."
        ),
        ReporterField.SCIENCE: (
            "Explain scientific discoveries and research clearly. "
            "Include quotes from researchers and scientists. "
            "Discuss implications for society and future research. "
            "Maintain scientific accuracy while being accessible. "
            "Cover peer-reviewed research and credible sources."
        ),
    }

    # Technology sub-section configurations
    TECHNOLOGY_SUB_SECTIONS: dict[TechnologySubSection, SubSectionConfig] = {
        TechnologySubSection.AI_TOOLS: SubSectionConfig(
            guidance=(
                "Focus specifically on practical AI applications and tools. "
                "MUST include specific tool names, companies, pricing, and version details. "
                "Cover user experiences, tool comparisons, feature reviews, and real-world implementation examples. "
                "Provide actionable information for readers evaluating these tools."
            ),
            search_terms="artificial intelligence AI tools software applications ChatGPT Claude Copilot productivity automation",
        ),
        TechnologySubSection.TECH_TRENDS: SubSectionConfig(
            guidance=(
                "Cover emerging technologies, industry shifts, startup ecosystem developments, future predictions, innovation patterns, disruptive technologies"
            ),
            search_terms="emerging technology innovation trends startups",
        ),
        TechnologySubSection.QUANTUM_COMPUTING: SubSectionConfig(
            guidance=(
                "Report on quantum research breakthroughs, commercial quantum applications, technical milestones, quantum hardware and software developments"
            ),
            search_terms="quantum computing qubits quantum hardware software",
        ),
        TechnologySubSection.GENERAL_TECH: SubSectionConfig(
            guidance=("Cover consumer technology, software updates, hardware releases, major tech company announcements, product launches"),
            search_terms="technology products software hardware releases",
        ),
        TechnologySubSection.MAJOR_DEALS: SubSectionConfig(
            guidance=("Focus on M&A activity, funding rounds, IPOs, strategic partnerships in the technology sector, venture capital trends"),
            search_terms="technology mergers acquisitions funding IPO investment",
        ),
    }

    # Economics sub-section configurations
    ECONOMICS_SUB_SECTIONS: dict[EconomicsSubSection, SubSectionConfig] = {
        EconomicsSubSection.CRYPTO: SubSectionConfig(
            guidance=("Cover blockchain developments, DeFi trends, regulatory changes, cryptocurrency market analysis, Web3 innovations, NFT markets"),
            search_terms="cryptocurrency blockchain bitcoin ethereum DeFi crypto market",
        ),
        EconomicsSubSection.US_STOCK_MARKET: SubSectionConfig(
            guidance=("Report on market movements, earnings reports, Federal Reserve policy, sector analysis, trading trends, market volatility"),
            search_terms="stock market NYSE NASDAQ trading S&P500 Dow Jones",
        ),
        EconomicsSubSection.GENERAL_NEWS: SubSectionConfig(
            guidance=("Cover global economic indicators, international trade, inflation data, employment statistics, economic policy, GDP reports"),
            search_terms="economy GDP inflation employment Federal Reserve economic policy",
        ),
        EconomicsSubSection.ISRAEL_ECONOMICS: SubSectionConfig(
            guidance=("Focus on Israeli tech sector, startup ecosystem, economic policy, innovation hub developments, regional economics, trade relationships"),
            search_terms="Israel economy Israeli tech startup ecosystem innovation hub",
        ),
    }

    # Science sub-section configurations
    SCIENCE_SUB_SECTIONS: dict[ScienceSubSection, SubSectionConfig] = {
        ScienceSubSection.NEW_RESEARCH: SubSectionConfig(
            guidance=("Cover breakthrough discoveries, peer-reviewed studies, research funding announcements, scientific innovations, paradigm shifts"),
            search_terms="scientific research breakthrough discovery study publication",
        ),
        ScienceSubSection.BIOLOGY: SubSectionConfig(
            guidance=("Report on life sciences, medical research, genetics breakthroughs, ecology, biotechnology developments, evolutionary discoveries"),
            search_terms="biology life sciences genetics biotech medical research",
        ),
        ScienceSubSection.CHEMISTRY: SubSectionConfig(
            guidance=("Cover materials science, pharmaceutical developments, chemical innovations, laboratory breakthroughs, molecular research"),
            search_terms="chemistry materials science pharmaceutical drug discovery",
        ),
        ScienceSubSection.PHYSICS: SubSectionConfig(
            guidance=("Report on fundamental research, space exploration, energy physics, theoretical and applied physics discoveries, particle physics"),
            search_terms="physics quantum mechanics space exploration particle physics",
        ),
    }

    @classmethod
    def get_field_guidelines(cls, field: ReporterField) -> str:
        """Get guidelines for a specific field."""
        return cls.FIELD_GUIDELINES.get(field, "Provide thorough, accurate reporting in your field")

    @classmethod
    def get_sub_section_config(cls, sub_section: TechnologySubSection | EconomicsSubSection | ScienceSubSection) -> SubSectionConfig | None:
        """Get configuration for a specific sub-section."""
        if isinstance(sub_section, TechnologySubSection):
            return cls.TECHNOLOGY_SUB_SECTIONS.get(sub_section)
        elif isinstance(sub_section, EconomicsSubSection):
            return cls.ECONOMICS_SUB_SECTIONS.get(sub_section)
        else:
            return cls.SCIENCE_SUB_SECTIONS.get(sub_section)

    @classmethod
    def get_sub_section_guidance(cls, sub_section: TechnologySubSection | EconomicsSubSection | ScienceSubSection) -> str:
        """Get guidance for a specific sub-section."""
        config = cls.get_sub_section_config(sub_section)
        return config.guidance if config else ""

    @classmethod
    def get_sub_section_search_terms(cls, sub_section: TechnologySubSection | EconomicsSubSection | ScienceSubSection) -> str:
        """Get search terms for a specific sub-section."""
        config = cls.get_sub_section_config(sub_section)
        return config.search_terms if config else ""


# ============================================================================
# PROMPT BUILDER CLASS
# ============================================================================


class ReporterPromptBuilder:
    """Builds prompts for reporter agent interactions."""

    def __init__(self, tool_registry: ReporterToolRegistry) -> None:
        """Initialize with tool registry for dynamic tool information."""
        self.tool_registry = tool_registry

    def create_system_prompt(self, field: ReporterField, sub_section: TechnologySubSection | EconomicsSubSection | ScienceSubSection | None = None) -> str:
        """Create the base system prompt for the reporter agent.

        Args:
            field: The field this reporter specializes in
            sub_section: Optional sub-section within the field

        Returns:
            System prompt string
        """
        # Base role description
        base_prompt = f"""You are an experienced {field.value} reporter for a major newspaper.
Your role is to research and write news stories in your field of expertise.

# FIELD-SPECIFIC GUIDELINES
{ReporterConfiguration.get_field_guidelines(field)}"""

        # Add sub-section specific guidance if available
        if sub_section:
            base_prompt += f"\n\n# SUB-SECTION FOCUS: {sub_section.value.upper()}\n"
            guidance = ReporterConfiguration.get_sub_section_guidance(sub_section)
            if guidance:
                base_prompt += f"When reporting on this sub-section: {guidance}"

        # Add general reporting standards and tool information

        base_prompt += f"""

# REPORTING STANDARDS
1. Write in an objective, journalistic style
2. Use the inverted pyramid structure (most important information first)
3. Include direct quotes when possible
4. Verify information from multiple sources
5. Provide context and background information
6. Use active voice and clear, concise language
7. Avoid bias and maintain journalistic integrity

# WORKFLOW PHASES
Your work follows a two-phase approach:

## PHASE 1: RESEARCH
- Use search and scrape tools to gather information
- Accumulate facts and sources about the topic
- Continue until you have sufficient reliable sources
- Maximum 4 research iterations allowed
- Return a ResearchResult when research is sufficient

## PHASE 2: WRITING
- Use accumulated research to write the story
- Base content on researched facts and sources
- Include all sources in the StoryDraft
- Return a complete StoryDraft

# AVAILABLE TOOLS

You have access to the following tools. Use the EXACT tool names shown below:

{self._format_tool_names_list()}

## Tool Details

{self.tool_registry.format_tools_for_prompt()}

## YouTube Tool Strategy

🎯 **IMPORTANT**: The youtube_search tool is your PRIMARY source for discovering trending topics in your field!

**Why YouTube channels matter:**
- Industry experts share insights before they hit mainstream news
- Real-time discussions about emerging trends and technologies
- Specialized content creators focus on your exact field/subsection
- Community reactions and expert analysis provide story angles

**When to use youtube_search:**
- **ALWAYS** for topic discovery tasks - use it FIRST before search tool
- For finding expert perspectives on breaking news
- To discover emerging trends in your specialized area
- To get technical insights and community reactions

**Your configured channels are field/subsection-specific:**
- Use your field and subsection parameters when calling youtube_search
- These channels are specifically curated for your reporting beat
- They provide insider knowledge that mainstream news sources miss

# OUTPUT FORMAT
You must respond with an AgentResponse object:
{AgentResponse.model_json_schema()}

The response must contain either:
1. A tool_call to execute a tool (with name and parameters)
2. A final response (story_draft, topic_list, or research_result based on task type)

# CRITICAL RULES
- You must either USE A TOOL or PROVIDE A FINAL ANSWER
- Never provide reasoning without taking action
- In research phase: use tools or return ResearchResult
- In writing phase: use tools or return StoryDraft
- Always respond with valid JSON matching the AgentResponse schema
- **IMPORTANT**: Only use the EXACT tool names listed above (e.g., `search`, `scrape`)
- **NEVER** make up tool names like "search_tool" or "scraper_tool"
- **ERROR HANDLING**: If you see tool execution failures in RECENT TOOL EXECUTIONS, fix your parameters based on the error messages
- **PARAMETER VALIDATION**: Always check error messages for parameter format issues (e.g., use 'url' not 'urls')
"""
        return base_prompt

    def build_task_prompt(self, state: ReporterState) -> str:
        """Build a complete prompt for the current task state.

        Args:
            state: Current reporter state

        Returns:
            Complete prompt for the LLM
        """
        task = state.current_task
        prompt_parts = [
            "# CURRENT TASK",
            f"Task Type: {task.name.value}",
            f"Field: {task.field.value}",
            f"Sub-Section: {task.sub_section.value if task.sub_section else 'General'}",
            f"Description: {task.description}",
            f"Iteration: {state.iteration}/{state.max_iterations}",
            "",
        ]

        # Add guidelines if provided (includes forbidden topics from editor)
        if task.guidelines:
            prompt_parts.extend([
                "# EDITORIAL GUIDELINES",
                task.guidelines,
                "",
            ])

        # Add task-specific instructions
        if task.name == TaskType.FIND_TOPICS:
            prompt_parts.extend(self._build_find_topics_instructions(task, state))
        elif task.name == TaskType.RESEARCH_TOPIC:
            prompt_parts.extend(self._build_research_topic_instructions(task, state))
        elif task.name == TaskType.WRITE_STORY:
            prompt_parts.extend(self._build_write_story_instructions(task, state))

        # Add state information
        prompt_parts.extend(["", "# CURRENT STATE", self._format_state_for_prompt(state)])

        return "\n".join(prompt_parts)

    def _build_find_topics_instructions(self, task: ReporterTask, state: ReporterState) -> list[str]:
        """Build instructions for find topics task."""
        instructions = [
            "# INSTRUCTIONS FOR FIND_TOPICS:",
            "",
            f"🎯 Your specialty: {task.field.value}" + (f" - {task.sub_section.value}" if task.sub_section else ""),
            "🎥 PRIORITY: Use youtube_search tool FIRST - your specialized channels have the best trending topics!",
            ""
        ]

        # Check what sources we have to determine next steps
        youtube_sources = [s for s in state.sources if 'youtube.com' in s.url]
        search_sources = [s for s in state.sources if 'youtube.com' not in s.url]

        if len(youtube_sources) > 0 and len(search_sources) > 0:
            # We have both YouTube and search sources - get topics from memory
            from agents.shared_memory_store import get_shared_memory_store
            memory_store = get_shared_memory_store()
            field_memories = memory_store.get_memories_by_field(task.field.value)
            available_topics = [memory.topic_name for memory in field_memories]

            instructions.extend(
                [
                    "",
                    "🎯 EXCELLENT! You have sources from BOTH YouTube and search tools.",
                    f"📊 YouTube sources: {len(youtube_sources)} | Search sources: {len(search_sources)}",
                    "",
                    "🚨 NOW return TopicList using EXACT topic names from memory!",
                    "",
                    f"Available topics stored in memory for {task.field.value}:",
                    *[f"  - {topic}" for topic in available_topics],
                    "",
                    f"Use these EXACT topic names (do not modify or rephrase them).",
                    f"Set field='{task.field.value}' and sub_section='{task.sub_section.value if task.sub_section else None}'.",
                    "",
                    "Return EXACTLY this structure:",
                    "{",
                    '  "reasoning": "Using exact topic names from SharedMemoryStore",',
                    '  "topic_list": {',
                    '    "reasoning": "Retrieved topics from memory with validated content", ',
                    f'    "topics": {available_topics[:5]},',  # Use actual topic names, limit to 5
                    f'    "field": "{task.field.value}",',
                    f'    "sub_section": "{task.sub_section.value if task.sub_section else None}"',
                    "  },",
                    f'  "iteration": {state.iteration},',
                    f'  "max_iterations": {state.max_iterations}',
                    "}",
                    "",
                    "DO NOT call tools. DO NOT provide anything else. Just return the TopicList.",
                ]
            )
        elif len(youtube_sources) > 0 and len(search_sources) == 0:
            # We have YouTube sources but no search sources - get search sources too
            instructions.extend([
                "",
                f"✅ Great! You have {len(youtube_sources)} YouTube sources with insider perspectives.",
                "🔍 Now use search tool to get broader news coverage for complete topic discovery.",
                '   - Use: {"query": "trending technology news today", "search_type": "news", "time_limit": "w"}',
                ""
            ])
        elif len(search_sources) > 0 and len(youtube_sources) == 0:
            # We have search sources but no YouTube sources - get YouTube sources too
            instructions.extend([
                "",
                f"✅ Good! You have {len(search_sources)} search sources with news coverage.",
                "🎥 Now use youtube_search tool to get specialized insights from your channels.",
                f'   - Use: {{"field": "{task.field.value}", "subsection": "{task.sub_section.value if task.sub_section else None}", "operation": "topics", "days_back": 7}}',
                ""
            ])
        else:
            # No search results yet, need to discover topics from multiple sources
            instructions.extend([
                "",
                "🔍 TOPIC DISCOVERY STRATEGY - Use BOTH sources for comprehensive coverage:",
                "",
                "🎯 STEP 1: Use youtube_search tool to find trending topics from your specialized channels",
                f'   - Use: {{"field": "{task.field.value}", "subsection": "{task.sub_section.value if task.sub_section else None}", "operation": "topics", "days_back": 7}}',
                "   - YouTube channels provide insider perspectives and emerging trends",
                "   - This gives you topics that mainstream news might miss",
                "",
                "🔍 STEP 2: Use search tool to find trending topics from news sources",
                '   - Use: {"query": "trending news today", "search_type": "news", "time_limit": "w"}',
                "   - This gives you broader market/news perspective",
                "",
                "🎯 STEP 3: Combine insights from both sources and return TopicList",
                "   - Prioritize topics that appear in both YouTube and news sources",
                "   - Include unique YouTube topics that show emerging trends"
            ])

        instructions.extend(["", "Expected TopicList schema:", f"{TopicList.model_json_schema()}"])

        return instructions

    def _build_research_topic_instructions(self, task: ReporterTask, state: ReporterState) -> list[str]:
        """Build instructions for research topic task."""
        instructions = [
            "# INSTRUCTIONS FOR RESEARCH_TOPIC:",
            f"Topic to research: {task.topic}",
        ]

        # Check current research progress
        if state.sources:
            instructions.extend(
                [
                    "",
                    f"PROGRESS: You have {len(state.sources)} sources (need {task.min_sources})",
                    f"Facts gathered: {len(state.accumulated_facts)}",
                ]
            )

            if len(state.sources) >= task.min_sources:
                instructions.extend(
                    [
                        "",
                        "✅ YOU HAVE ENOUGH SOURCES! Now return a ResearchResult:",
                        "- Summarize the key findings",
                        "- Include all facts you've gathered",
                        "- List all sources",
                        "- Provide key points for the story",
                    ]
                )
            else:
                instructions.extend(
                    [
                        "",
                        f"⚠️ Need {task.min_sources - len(state.sources)} more sources!",
                        "Continue researching:",
                        "- Use search tool for more recent news",
                        "- Use scrape tool on promising URLs",
                    ]
                )
        else:
            instructions.extend(
                [
                    "",
                    "START RESEARCHING:",
                    "1. Use search tool with 'news' search_type",
                    "2. Use scrape tool to get detailed facts",
                    f"3. Gather at least {task.min_sources} credible sources",
                ]
            )

        instructions.extend(["", "Expected ResearchResult schema:", f"{ResearchResult.model_json_schema()}"])

        return instructions

    def _build_write_story_instructions(self, task: ReporterTask, state: ReporterState) -> list[str]:
        """Build instructions for write story task."""
        # Check if we have memory sources available
        memory_sources_available = len(state.sources) > 0

        if memory_sources_available:
            # If memory sources are available, instruct to use them directly
            return [
                "# MEMORY SOURCES AVAILABLE - WRITE STORY DIRECTLY",
                f"Topic: {task.topic}",
                f"📚 Available Sources: {len(state.sources)} sources from previous research",
                "",
                "🚨 IMPORTANT: You have sufficient sources from memory - DO NOT perform additional research!",
                "Use the existing sources below to write your story immediately.",
                "",
                "WRITING INSTRUCTIONS:",
                "1. Use the research sources below to write a comprehensive story",
                f"2. Target word count: {task.target_word_count} words",
                "3. Include proper attribution and quotes where relevant",
                "4. Structure: headline, lead paragraph, body paragraphs, conclusion",
                "5. Ensure all sources are properly referenced",
                "",
                "CRITICAL: Return a StoryDraft with the complete article - NO additional research needed!",
                "",
                "Expected StoryDraft schema:",
                f"{StoryDraft.model_json_schema()}",
            ]

        elif state.task_phase == TaskPhase.RESEARCH:
            # Research phase instructions - check memory first, then search
            from agents.shared_memory_store import get_shared_memory_store
            memory_store = get_shared_memory_store()
            available_memory_topics = memory_store.list_topics(field=task.field.value)

            warning_msg = ""
            if state.research_iteration_count >= 2:
                warning_msg = f"\n⚠️  WARNING: Research iteration {state.research_iteration_count + 1}/4 - Task will FAIL if no sources are found!"

            instructions = [
                "# CURRENT PHASE: RESEARCH",
                f"Topic: {task.topic}",
                f"Research Progress: {len(state.sources)}/{task.min_sources} sources found",
                f"Research Iteration: {state.research_iteration_count + 1}/4 (max 4 iterations){warning_msg}",
                "",
                "🧠 MEMORY-FIRST RESEARCH STRATEGY:",
            ]

            if available_memory_topics:
                instructions.extend([
                    f"Available topics in SharedMemoryStore for {task.field.value}:",
                    *[f"  - '{topic}'" for topic in available_memory_topics],
                    "",
                    "1. FIRST: Check if your topic matches any memory topic above",
                    f"   - If '{task.topic}' is similar to any memory topic, use fetch_from_memory tool",
                    f'   - Use: {{"topic_key": "exact_memory_topic_name", "field": "{task.field.value}"}}',
                    "   - This gives you pre-validated research content instantly",
                    "",
                    "2. THEN: If no memory match or need more content, use search/scrape tools",
                    f"   - Search for: '{task.topic}'",
                    "   - Scrape promising URLs for detailed content",
                    "",
                ])
            else:
                instructions.extend([
                    "No topics found in memory - use traditional research methods:",
                    "",
                ])

            instructions.extend([
                "RESEARCH INSTRUCTIONS:",
                f"🥇 PRIORITY: Use fetch_from_memory if topic matches memory",
                f"🥈 FALLBACK: Search EXACTLY for: '{task.topic}' (use this exact text as query)",
                "   - Use search tool with 'news' search_type and 'time_limit': 'w'",
                "   - This will give you source URLs to investigate",
                "",
                "🥉 DETAILED: Use scrape tool to get detailed content from EACH promising source URL",
                "   - Scrape at least 2-3 URLs from your search results",
                '   - Use: {"url": "single_url_here"} format (not arrays!)',
                "   - Look for facts, quotes, statistics, company names, dates, numbers",
                "",
                f"🎯 COLLECT: Gather at least {task.min_sources} reliable sources with detailed facts",
                "📝 RETURN: Once you have sufficient facts, return a ResearchResult",
                "",
                "CRITICAL: You must either use a tool OR return a ResearchResult",
                "",
                "Expected ResearchResult schema:",
                f"{ResearchResult.model_json_schema()}",
            ])

            return instructions
        else:
            # Writing phase instructions
            return [
                "# CURRENT PHASE: STORY WRITING",
                f"Topic: {task.topic}",
                f"Available Research: {len(state.sources)} sources, {len(state.accumulated_facts)} facts",
                "",
                "WRITING INSTRUCTIONS:",
                "1. Use the research facts and sources to write a comprehensive story",
                f"2. MANDATORY: Write at least {max(task.target_word_count, 500)} words",
                f"3. Target word count: {task.target_word_count} words",
                "4. Base your story on the research facts you've gathered",
                "5. CRITICAL: Include ALL research sources in the StoryDraft",
                "6. Ensure the story is comprehensive and well-structured",
                "",
                "AVAILABLE RESEARCH FACTS:",
                *[f"- {fact}" for fact in state.accumulated_facts[:10]],
                "" if len(state.accumulated_facts) <= 10 else f"... and {len(state.accumulated_facts) - 10} more facts",
                "",
                "IMPORTANT: You must either use a tool OR return a StoryDraft",
                "",
                "Expected StoryDraft schema:",
                f"{StoryDraft.model_json_schema()}",
            ]

    def _format_state_for_prompt(self, state: ReporterState) -> str:
        """Format COMPLETE state for inclusion in prompt.

        Args:
            state: Current reporter state

        Returns:
            Complete state dump as formatted string
        """

        # Convert state to a comprehensive dictionary for display
        state_info = {
            "basic_info": {
                "phase": state.task_phase.value,
                "iteration": f"{state.iteration}/{state.max_iterations}",
                "research_iteration": f"{state.research_iteration_count}/4",
            },
            "task_info": {
                "name": state.current_task.name.value,
                "field": state.current_task.field.value,
                "sub_section": state.current_task.sub_section.value if state.current_task.sub_section else None,
                "topic": getattr(state.current_task, "topic", "N/A"),
                "description": state.current_task.description,
            },
            "progress": {
                "sources_collected": len(state.sources),
                "facts_accumulated": len(state.accumulated_facts),
                "tool_calls_made": len(state.tool_calls),
                "tool_results_logged": len(state.tool_results),
                "errors_logged": len(state.errors),
            },
        }

        # Format as readable sections
        output_parts = [
            "# COMPLETE STATE DUMP",
            "",
            f"📋 BASIC INFO: Phase={state_info['basic_info']['phase']}, Iter={state_info['basic_info']['iteration']}, Research={state_info['basic_info']['research_iteration']}",
            f"🎯 TASK: {state_info['task_info']['name']} | Field={state_info['task_info']['field']} | Sub={state_info['task_info']['sub_section']}",
            f"📊 PROGRESS: Sources={state_info['progress']['sources_collected']}, Facts={state_info['progress']['facts_accumulated']}, Tools={state_info['progress']['tool_calls_made']}, Results={state_info['progress']['tool_results_logged']}, Errors={state_info['progress']['errors_logged']}",
        ]

        # ALL TOOL EXECUTIONS (most important for debugging)
        if state.tool_results:
            output_parts.append(f"\n🔧 ALL TOOL EXECUTIONS ({len(state.tool_results)}):")
            for i, tool_result in enumerate(state.tool_results, 1):
                status = "✅" if tool_result.success else "❌"
                output_parts.append(f"  {i}. {status} {tool_result.tool_name} (iter:{tool_result.iteration})")
                if tool_result.error:
                    output_parts.append(f"     ⚠️  ERROR: {tool_result.error}")
                if not tool_result.success and tool_result.result:
                    preview = tool_result.result[:150] + "..." if len(tool_result.result) > 150 else tool_result.result
                    output_parts.append(f"     📝 DETAILS: {preview}")

        # ALL TOOL CALLS (what was requested)
        if state.tool_calls:
            output_parts.append(f"\n📞 ALL TOOL CALLS ({len(state.tool_calls)}):")
            for i, tool_call in enumerate(state.tool_calls, 1):
                params_str = str(tool_call.parameters)[:100] + "..." if len(str(tool_call.parameters)) > 100 else str(tool_call.parameters)
                output_parts.append(f"  {i}. {tool_call.name} → {params_str}")

        # ALL ERRORS
        if state.errors:
            output_parts.append(f"\n🚨 ALL ERRORS ({len(state.errors)}):")
            for i, error in enumerate(state.errors, 1):
                output_parts.append(f"  {i}. {error}")

        # ALL SOURCES
        if state.sources:
            output_parts.append(f"\n📚 ALL SOURCES ({len(state.sources)}):")
            for i, source in enumerate(state.sources, 1):
                output_parts.append(f"  {i}. {source.title}")
                output_parts.append(f"     🔗 {source.url}")
                if source.summary:
                    summary_preview = source.summary[:100] + "..." if len(source.summary) > 100 else source.summary
                    output_parts.append(f"     📄 {summary_preview}")

        # ALL FACTS
        if state.accumulated_facts:
            output_parts.append(f"\n💡 ALL FACTS ({len(state.accumulated_facts)}):")
            for i, fact in enumerate(state.accumulated_facts, 1):
                fact_preview = fact[:150] + "..." if len(fact) > 150 else fact
                output_parts.append(f"  {i}. {fact_preview}")

        # ALL SEARCH HISTORY
        if state.search_results:
            output_parts.append(f"\n🔍 ALL SEARCHES ({len(state.search_results)}):")
            for i, search in enumerate(state.search_results, 1):
                output_parts.append(f"  {i}. '{search.query}' → {search.result_count} results")

        # CURRENT RESEARCH (if available)
        if state.current_research:
            output_parts.append("\n🔬 CURRENT RESEARCH:")
            output_parts.append(f"  📝 Summary: {state.current_research.summary[:200]}...")
            output_parts.append(f"  🔑 Key Points: {len(state.current_research.key_points)}")
            output_parts.append(f"  📚 Sources: {len(state.current_research.sources)}")

        # COMMAND HISTORY
        if state.command_history:
            output_parts.append(f"\n📋 COMMAND HISTORY ({len(state.command_history)}):")
            for i, cmd in enumerate(state.command_history, 1):
                output_parts.append(f"  {i}. {cmd}")

        return "\n".join(output_parts)

    def _format_tool_names_list(self) -> str:
        """Format a simple list of available tool names.

        Returns:
            Formatted list of tool names
        """
        tool_names = list(self.tool_registry.tools.keys())
        return "- " + "\n- ".join(f"`{name}`" for name in tool_names)
