"""Strain Knowledge Agent for CNApy Multi-Agent System

This agent provides LLM-based strain knowledge queries, including:
- Reaction/gene existence analysis in specific strains
- Strain metabolism characteristics
- Strain comparison
- Metabolic engineering suggestions
- Literature-based information search

It integrates with the existing LLMConfig system for multi-provider support.
"""

import hashlib
import json
import re
import time
from pathlib import Path

from cnapy.agents.base_agent import (
    AgentContext,
    BaseAgent,
    Skill,
    SkillResult,
    SkillStatus,
    ToolDefinition,
)
from cnapy.gui_elements.llm_analysis_dialog import LLMConfig


class StrainKnowledgeAgent(BaseAgent):
    """Agent for LLM-based strain knowledge queries.

    This agent provides expertise in:
    - Strain-specific reaction and gene analysis
    - Metabolic characteristics of different organisms
    - Comparative strain analysis
    - Metabolic engineering recommendations
    """

    name = "StrainKnowledgeAgent"
    description = "LLM 기반 균주 지식 전문가 / LLM-based strain knowledge expert"

    # Keywords that trigger this agent
    TRIGGER_KEYWORDS = [
        # English
        "strain",
        "organism",
        "species",
        "bacteria",
        "yeast",
        "exist",
        "presence",
        "ortholog",
        "homolog",
        "pathway",
        "metabolism",
        "metabolic",
        "literature",
        "reference",
        "paper",
        "compare",
        "comparison",
        "difference",
        "modification",
        "engineering",
        "suggest",
        # Korean
        "균주",
        "생물",
        "종",
        "박테리아",
        "효모",
        "존재",
        "유무",
        "오솔로그",
        "호몰로그",
        "경로",
        "대사",
        "대사체",
        "문헌",
        "참고",
        "논문",
        "비교",
        "차이",
        "개량",
        "엔지니어링",
        "제안",
    ]

    def __init__(self, context: AgentContext, llm_config: LLMConfig | None = None):
        super().__init__(context)
        self.llm_config = llm_config or LLMConfig()
        self._register_skills()

    def _register_skills(self):
        """Register all skills for this agent."""
        # Skill: Analyze strain reactions
        self.register_skill(
            Skill(
                name="analyze_strain_reactions",
                description="균주에서 특정 반응의 존재 여부 분석 / Analyze reaction existence in a strain",
                parameters={
                    "strain_name": {"type": "string", "description": "Strain name (e.g., 'E. coli', 'C. glutamicum')"},
                    "reaction_ids": {"type": "array", "description": "List of reaction IDs to analyze"},
                },
                required_params=["strain_name"],
                handler=self._analyze_strain_reactions,
            )
        )

        # Skill: Analyze strain genes
        self.register_skill(
            Skill(
                name="analyze_strain_genes",
                description="균주에서 특정 유전자의 존재/오솔로그 분석 / Analyze gene/ortholog existence in a strain",
                parameters={
                    "strain_name": {"type": "string", "description": "Strain name"},
                    "gene_ids": {"type": "array", "description": "List of gene IDs to analyze"},
                },
                required_params=["strain_name"],
                handler=self._analyze_strain_genes,
            )
        )

        # Skill: Get strain metabolism
        self.register_skill(
            Skill(
                name="get_strain_metabolism",
                description="균주의 대사 특성 조회 / Get strain metabolic characteristics",
                parameters={
                    "strain_name": {"type": "string", "description": "Strain name"},
                    "aspect": {
                        "type": "string",
                        "description": "Specific aspect (e.g., 'carbon_sources', 'respiration', 'fermentation')",
                    },
                },
                required_params=["strain_name"],
                handler=self._get_strain_metabolism,
            )
        )

        # Skill: Compare strains
        self.register_skill(
            Skill(
                name="compare_strains",
                description="두 균주 간 대사 비교 / Compare metabolism between two strains",
                parameters={
                    "strain1": {"type": "string", "description": "First strain name"},
                    "strain2": {"type": "string", "description": "Second strain name"},
                    "focus": {
                        "type": "string",
                        "description": "Comparison focus (e.g., 'general', 'pathways', 'genes')",
                    },
                },
                required_params=["strain1", "strain2"],
                handler=self._compare_strains,
            )
        )

        # Skill: Suggest modifications
        self.register_skill(
            Skill(
                name="suggest_modifications",
                description="목표 물질 생산을 위한 대사공학 전략 제안 / Suggest metabolic engineering strategies",
                parameters={
                    "strain_name": {"type": "string", "description": "Base strain name"},
                    "target_product": {"type": "string", "description": "Target product to produce"},
                    "constraints": {"type": "array", "description": "Any constraints or preferences"},
                },
                required_params=["strain_name", "target_product"],
                handler=self._suggest_modifications,
            )
        )

        # Skill: Literature search
        self.register_skill(
            Skill(
                name="literature_search",
                description="문헌 기반 정보 검색 / Literature-based information search",
                parameters={
                    "query": {"type": "string", "description": "Search query"},
                    "strain_context": {"type": "string", "description": "Strain context for the search"},
                },
                required_params=["query"],
                handler=self._literature_search,
            )
        )

        # Skill: Check reaction in strain
        self.register_skill(
            Skill(
                name="check_reaction_in_strain",
                description="균주에 특정 반응이 있는지 확인 / Check if a reaction exists in a strain",
                parameters={
                    "strain_name": {"type": "string", "description": "Strain name"},
                    "reaction_id": {"type": "string", "description": "Reaction ID to check"},
                },
                required_params=["strain_name", "reaction_id"],
                handler=self._check_reaction_in_strain,
            )
        )

        # Skill: Check gene in strain
        self.register_skill(
            Skill(
                name="check_gene_in_strain",
                description="균주에 특정 유전자가 있는지 확인 / Check if a gene exists in a strain",
                parameters={
                    "strain_name": {"type": "string", "description": "Strain name"},
                    "gene_id": {"type": "string", "description": "Gene ID to check"},
                },
                required_params=["strain_name", "gene_id"],
                handler=self._check_gene_in_strain,
            )
        )

    def get_tools(self) -> list[ToolDefinition]:
        """Return tool definitions for LLM function calling."""
        tools = []
        for skill in self.skills.values():
            tools.append(ToolDefinition(name=skill.name, description=skill.description, parameters=skill.parameters))
        return tools

    def can_handle(self, intent: str) -> float:
        """Calculate ability to handle the given intent."""
        intent_lower = intent.lower()
        score = 0.0

        # Check for trigger keywords
        keyword_matches = sum(1 for kw in self.TRIGGER_KEYWORDS if kw in intent_lower)
        if keyword_matches > 0:
            score += min(0.3 + (keyword_matches * 0.1), 0.6)

        # Check for strain-specific patterns
        strain_patterns = [
            r"\b(e\.?\s*coli|escherichia)\b",
            r"\b(c\.?\s*glutamicum|corynebacterium)\b",
            r"\b(s\.?\s*cerevisiae|saccharomyces)\b",
            r"\b(b\.?\s*subtilis|bacillus)\b",
            r"\b균주\s*\S+\b",
            r"\b\S+에\s*(있|존재)",
        ]
        for pattern in strain_patterns:
            if re.search(pattern, intent_lower, re.IGNORECASE):
                score += 0.2
                break

        # Check for LLM-requiring patterns (knowledge queries)
        llm_patterns = [
            r"(있나요|있어요|있습니까|존재하나요)",
            r"(does|exist|have|presence)",
            r"(ortholog|homolog|similar)",
            r"(compare|difference|versus|vs)",
            r"(suggest|recommend|strategy)",
        ]
        for pattern in llm_patterns:
            if re.search(pattern, intent_lower, re.IGNORECASE):
                score += 0.15

        return min(score, 1.0)

    def _get_api_key(self) -> str:
        """Get the API key for the configured provider."""
        provider = self.llm_config.default_provider
        if provider == "openai":
            return self.llm_config.openai_api_key
        elif provider == "anthropic":
            return self.llm_config.anthropic_api_key
        else:  # gemini
            return self.llm_config.gemini_api_key

    def _get_model(self) -> str:
        """Get the model name for the configured provider."""
        provider = self.llm_config.default_provider
        if provider == "openai":
            return self.llm_config.default_model_openai
        elif provider == "anthropic":
            return self.llm_config.default_model_anthropic
        else:  # gemini
            return self.llm_config.default_model_gemini

    def _get_cache_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        key_str = ":".join(str(arg) for arg in args)
        key_str += f":{self.llm_config.default_provider}:{self._get_model()}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> dict | None:
        """Get cached result if valid."""
        if not self.llm_config.use_cache:
            return None

        cache_file = Path(self.llm_config.cache_dir) / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    data = json.load(f)
                    timestamp = data.get("timestamp", 0)
                    expiry_seconds = self.llm_config.cache_expiry_days * 86400
                    if time.time() - timestamp < expiry_seconds:
                        return data.get("result")
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _save_to_cache(self, cache_key: str, result: dict):
        """Save result to cache."""
        if not self.llm_config.use_cache:
            return

        cache_dir = Path(self.llm_config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"timestamp": time.time(), "result": result}, f, ensure_ascii=False)
        except OSError:
            pass

    def _call_llm(self, prompt: str, system_prompt: str | None = None) -> str:
        """Call the LLM API with the given prompt."""
        provider = self.llm_config.default_provider

        if not system_prompt:
            system_prompt = (
                "You are a bioinformatics expert specializing in metabolic network analysis, "
                "comparative genomics, and metabolic engineering. Provide accurate, evidence-based "
                "analysis. When uncertain, clearly indicate the level of confidence."
            )

        if provider == "openai":
            return self._call_openai(prompt, system_prompt)
        elif provider == "anthropic":
            return self._call_anthropic(prompt, system_prompt)
        else:  # gemini
            return self._call_gemini(prompt, system_prompt)

    def _call_openai(self, prompt: str, system_prompt: str) -> str:
        """Call OpenAI API."""
        try:
            import openai
        except ImportError as err:
            raise ImportError("OpenAI package not installed. Install with: pip install openai") from err

        api_key = self.llm_config.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key not configured")

        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=self._get_model(),
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str, system_prompt: str) -> str:
        """Call Anthropic Claude API."""
        try:
            from anthropic import Anthropic
        except ImportError as err:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic") from err

        api_key = self.llm_config.anthropic_api_key
        if not api_key:
            raise ValueError("Anthropic API key not configured")

        client = Anthropic(api_key=api_key)

        message = client.messages.create(
            model=self._get_model(),
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text

    def _call_gemini(self, prompt: str, system_prompt: str) -> str:
        """Call Google Gemini API."""
        try:
            import google.generativeai as genai
        except ImportError as err:
            raise ImportError(
                "Google Generative AI package not installed. Install with: pip install google-generativeai"
            ) from err

        api_key = self.llm_config.gemini_api_key
        if not api_key:
            raise ValueError("Gemini API key not configured")

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name=self._get_model(),
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 2048,
            },
        )

        full_prompt = f"{system_prompt}\n\n{prompt}"
        response = model.generate_content(full_prompt)

        return response.text

    def _parse_json_response(self, content: str) -> dict:
        """Parse JSON from LLM response."""
        # Try to find JSON in markdown code block
        code_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find balanced JSON object
        start = content.find("{")
        if start != -1:
            depth = 0
            in_string = False
            escape_next = False

            for i, c in enumerate(content[start:], start):
                if escape_next:
                    escape_next = False
                    continue
                if c == "\\" and in_string:
                    escape_next = True
                    continue
                if c == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(content[start : i + 1])
                            except json.JSONDecodeError:
                                pass
                            break

        # Fallback: return content as raw response
        return {"raw_response": content, "parsed": False}

    # =========================================================================
    # Skill Handlers
    # =========================================================================

    def _analyze_strain_reactions(self, strain_name: str = "", reaction_ids: list[str] = None, **kwargs) -> SkillResult:
        """Analyze reaction existence in a strain."""
        if reaction_ids is None:
            reaction_ids = []

        if not strain_name:
            return SkillResult(
                status=SkillStatus.FAILURE, message="Strain name is required", message_ko="균주 이름이 필요합니다"
            )

        # If no specific reactions, use model reactions
        if not reaction_ids:
            model = self.context.appdata.project.cobra_py_model
            if model:
                # Get a sample of reactions for analysis
                reaction_ids = [r.id for r in list(model.reactions)[:10]]
            else:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    message="No model loaded and no reactions specified",
                    message_ko="모델이 로드되지 않았고 반응이 지정되지 않았습니다",
                )

        # Check cache
        cache_key = self._get_cache_key("reactions", strain_name, ",".join(reaction_ids[:10]))
        cached = self._get_cached_result(cache_key)
        if cached:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Analysis of reactions in {strain_name} (cached)",
                message_ko=f"{strain_name}의 반응 분석 (캐시됨)",
                data=cached,
            )

        # Build prompt
        reaction_list = "\n".join([f"- {rid}" for rid in reaction_ids[:20]])
        prompt = f"""Analyze whether the following metabolic reactions exist or are likely to exist in {strain_name}.

Reactions to analyze:
{reaction_list}

For each reaction, provide:
1. Existence status (Yes/No/Unknown/Likely)
2. Confidence level (High/Medium/Low)
3. Brief evidence or reasoning

Respond in JSON format:
{{
    "strain": "{strain_name}",
    "reactions": {{
        "reaction_id": {{
            "exists": "Yes/No/Unknown/Likely",
            "confidence": "High/Medium/Low",
            "evidence": "brief explanation"
        }}
    }},
    "summary": "overall summary of findings"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            self._save_to_cache(cache_key, result)

            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Analyzed {len(reaction_ids)} reactions in {strain_name}",
                message_ko=f"{strain_name}에서 {len(reaction_ids)}개 반응 분석 완료",
                data=result,
            )
        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE, message=f"LLM API error: {str(e)}", message_ko=f"LLM API 오류: {str(e)}"
            )

    def _analyze_strain_genes(self, strain_name: str = "", gene_ids: list[str] = None, **kwargs) -> SkillResult:
        """Analyze gene existence in a strain."""
        if gene_ids is None:
            gene_ids = []

        if not strain_name:
            return SkillResult(
                status=SkillStatus.FAILURE, message="Strain name is required", message_ko="균주 이름이 필요합니다"
            )

        # If no specific genes, use model genes
        if not gene_ids:
            model = self.context.appdata.project.cobra_py_model
            if model:
                gene_ids = [g.id for g in list(model.genes)[:10]]
            else:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    message="No model loaded and no genes specified",
                    message_ko="모델이 로드되지 않았고 유전자가 지정되지 않았습니다",
                )

        # Check cache
        cache_key = self._get_cache_key("genes", strain_name, ",".join(gene_ids[:10]))
        cached = self._get_cached_result(cache_key)
        if cached:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Analysis of genes in {strain_name} (cached)",
                message_ko=f"{strain_name}의 유전자 분석 (캐시됨)",
                data=cached,
            )

        # Build prompt
        gene_list = "\n".join([f"- {gid}" for gid in gene_ids[:20]])
        prompt = f"""Analyze whether the following genes exist or have orthologs in {strain_name}.

Genes to analyze:
{gene_list}

For each gene, provide:
1. Existence status (Yes/No/Unknown/Likely)
2. Confidence level (High/Medium/Low)
3. Ortholog name if different
4. Brief evidence or reasoning

Respond in JSON format:
{{
    "strain": "{strain_name}",
    "genes": {{
        "gene_id": {{
            "exists": "Yes/No/Unknown/Likely",
            "confidence": "High/Medium/Low",
            "ortholog": "ortholog name or null",
            "evidence": "brief explanation"
        }}
    }},
    "summary": "overall summary of findings"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            self._save_to_cache(cache_key, result)

            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Analyzed {len(gene_ids)} genes in {strain_name}",
                message_ko=f"{strain_name}에서 {len(gene_ids)}개 유전자 분석 완료",
                data=result,
            )
        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE, message=f"LLM API error: {str(e)}", message_ko=f"LLM API 오류: {str(e)}"
            )

    def _get_strain_metabolism(self, strain_name: str = "", aspect: str = "general", **kwargs) -> SkillResult:
        """Get strain metabolic characteristics."""

        if not strain_name:
            return SkillResult(
                status=SkillStatus.FAILURE, message="Strain name is required", message_ko="균주 이름이 필요합니다"
            )

        # Check cache
        cache_key = self._get_cache_key("metabolism", strain_name, aspect)
        cached = self._get_cached_result(cache_key)
        if cached:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Metabolic characteristics of {strain_name} (cached)",
                message_ko=f"{strain_name}의 대사 특성 (캐시됨)",
                data=cached,
            )

        aspect_prompts = {
            "general": "overall metabolic capabilities and characteristics",
            "carbon_sources": "carbon source utilization capabilities",
            "respiration": "respiration and electron transport characteristics",
            "fermentation": "fermentation capabilities and products",
            "amino_acids": "amino acid biosynthesis and requirements",
            "cofactors": "cofactor biosynthesis and requirements",
        }

        focus = aspect_prompts.get(aspect, aspect_prompts["general"])

        prompt = f"""Provide detailed information about the {focus} of {strain_name}.

Include:
1. Key metabolic features
2. Notable pathways present or absent
3. Industrial relevance if any
4. Common genetic modifications for metabolic engineering

Respond in JSON format:
{{
    "strain": "{strain_name}",
    "aspect": "{aspect}",
    "characteristics": {{
        "key_features": ["feature1", "feature2"],
        "pathways": {{
            "present": ["pathway1", "pathway2"],
            "absent": ["pathway3", "pathway4"]
        }},
        "industrial_applications": ["application1", "application2"],
        "common_modifications": ["modification1", "modification2"]
    }},
    "summary": "brief summary"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            self._save_to_cache(cache_key, result)

            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Retrieved metabolic characteristics of {strain_name}",
                message_ko=f"{strain_name}의 대사 특성 조회 완료",
                data=result,
            )
        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE, message=f"LLM API error: {str(e)}", message_ko=f"LLM API 오류: {str(e)}"
            )

    def _compare_strains(self, strain1: str = "", strain2: str = "", focus: str = "general", **kwargs) -> SkillResult:
        """Compare metabolism between two strains."""

        if not strain1 or not strain2:
            return SkillResult(
                status=SkillStatus.FAILURE,
                message="Two strain names are required",
                message_ko="두 균주 이름이 필요합니다",
            )

        # Check cache
        cache_key = self._get_cache_key("compare", strain1, strain2, focus)
        cached = self._get_cached_result(cache_key)
        if cached:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Comparison of {strain1} and {strain2} (cached)",
                message_ko=f"{strain1}과 {strain2} 비교 (캐시됨)",
                data=cached,
            )

        prompt = f"""Compare the metabolic characteristics of {strain1} and {strain2}.

Focus area: {focus}

Provide:
1. Key similarities
2. Key differences
3. Unique features of each strain
4. Implications for metabolic engineering

Respond in JSON format:
{{
    "strain1": "{strain1}",
    "strain2": "{strain2}",
    "focus": "{focus}",
    "comparison": {{
        "similarities": ["similarity1", "similarity2"],
        "differences": [
            {{
                "aspect": "aspect name",
                "{strain1}": "characteristic in strain1",
                "{strain2}": "characteristic in strain2"
            }}
        ],
        "unique_features": {{
            "{strain1}": ["feature1", "feature2"],
            "{strain2}": ["feature1", "feature2"]
        }}
    }},
    "engineering_implications": "implications for choosing between strains",
    "summary": "brief comparison summary"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            self._save_to_cache(cache_key, result)

            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Compared {strain1} and {strain2}",
                message_ko=f"{strain1}과 {strain2} 비교 완료",
                data=result,
            )
        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE, message=f"LLM API error: {str(e)}", message_ko=f"LLM API 오류: {str(e)}"
            )

    def _suggest_modifications(
        self, strain_name: str = "", target_product: str = "", constraints: list[str] = None, **kwargs
    ) -> SkillResult:
        """Suggest metabolic engineering strategies."""
        if constraints is None:
            constraints = []

        if not strain_name or not target_product:
            return SkillResult(
                status=SkillStatus.FAILURE,
                message="Strain name and target product are required",
                message_ko="균주 이름과 목표 물질이 필요합니다",
            )

        # Check cache
        cache_key = self._get_cache_key("suggest", strain_name, target_product, ",".join(constraints))
        cached = self._get_cached_result(cache_key)
        if cached:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Engineering suggestions for {target_product} (cached)",
                message_ko=f"{target_product} 생산을 위한 제안 (캐시됨)",
                data=cached,
            )

        constraints_text = "\n".join([f"- {c}" for c in constraints]) if constraints else "None specified"

        prompt = f"""Suggest metabolic engineering strategies to produce {target_product} in {strain_name}.

Constraints/preferences:
{constraints_text}

Provide:
1. Key gene knockouts recommended
2. Gene overexpressions recommended
3. Heterologous genes to introduce
4. Pathway modifications
5. Culture condition recommendations

Respond in JSON format:
{{
    "strain": "{strain_name}",
    "target_product": "{target_product}",
    "strategies": {{
        "knockouts": [
            {{"gene": "gene_id", "reason": "explanation"}}
        ],
        "overexpressions": [
            {{"gene": "gene_id", "reason": "explanation"}}
        ],
        "heterologous_genes": [
            {{"gene": "gene_id", "source": "source organism", "reason": "explanation"}}
        ],
        "pathway_modifications": ["modification1", "modification2"],
        "culture_conditions": {{
            "carbon_source": "recommended carbon source",
            "oxygen": "aerobic/anaerobic/microaerobic",
            "other": "other recommendations"
        }}
    }},
    "expected_challenges": ["challenge1", "challenge2"],
    "references": ["relevant papers or resources"],
    "summary": "brief strategy summary"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            self._save_to_cache(cache_key, result)

            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Generated engineering suggestions for {target_product} production",
                message_ko=f"{target_product} 생산을 위한 대사공학 전략 제안 완료",
                data=result,
            )
        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE, message=f"LLM API error: {str(e)}", message_ko=f"LLM API 오류: {str(e)}"
            )

    def _literature_search(self, query: str = "", strain_context: str = "", **kwargs) -> SkillResult:
        """Literature-based information search."""

        if not query:
            return SkillResult(
                status=SkillStatus.FAILURE, message="Search query is required", message_ko="검색어가 필요합니다"
            )

        # Check cache
        cache_key = self._get_cache_key("literature", query, strain_context)
        cached = self._get_cached_result(cache_key)
        if cached:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message="Literature search results (cached)",
                message_ko="문헌 검색 결과 (캐시됨)",
                data=cached,
            )

        context_text = f" in the context of {strain_context}" if strain_context else ""

        prompt = f"""Provide a literature-based summary about: {query}{context_text}

Include:
1. Key findings from scientific literature
2. Important studies and their conclusions
3. Current state of knowledge
4. Open questions or controversies

Respond in JSON format:
{{
    "query": "{query}",
    "context": "{strain_context}",
    "findings": [
        {{
            "topic": "topic name",
            "summary": "brief summary",
            "key_studies": ["study description 1", "study description 2"]
        }}
    ],
    "current_state": "summary of current knowledge",
    "open_questions": ["question1", "question2"],
    "relevant_databases": ["database1", "database2"],
    "summary": "overall summary"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            self._save_to_cache(cache_key, result)

            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Literature search completed for: {query}",
                message_ko=f"문헌 검색 완료: {query}",
                data=result,
            )
        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE, message=f"LLM API error: {str(e)}", message_ko=f"LLM API 오류: {str(e)}"
            )

    def _check_reaction_in_strain(self, strain_name: str = "", reaction_id: str = "", **kwargs) -> SkillResult:
        """Check if a specific reaction exists in a strain."""

        if not strain_name or not reaction_id:
            return SkillResult(
                status=SkillStatus.FAILURE,
                message="Strain name and reaction ID are required",
                message_ko="균주 이름과 반응 ID가 필요합니다",
            )

        # Get reaction name from model if available
        reaction_name = reaction_id
        model = self.context.appdata.project.cobra_py_model
        if model and reaction_id in model.reactions:
            rxn = model.reactions.get_by_id(reaction_id)
            reaction_name = rxn.name or reaction_id

        # Check cache
        cache_key = self._get_cache_key("check_reaction", strain_name, reaction_id)
        cached = self._get_cached_result(cache_key)
        if cached:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Reaction {reaction_id} in {strain_name} (cached)",
                message_ko=f"{strain_name}의 {reaction_id} 반응 (캐시됨)",
                data=cached,
            )

        prompt = f"""Analyze whether the following metabolic reaction exists in {strain_name}.

Reaction ID: {reaction_id}
Reaction Name: {reaction_name}

Please provide:
1. Does this reaction exist in {strain_name}? (Yes/No/Unknown/Likely)
2. How confident is this assessment? (High/Medium/Low)
3. What evidence supports this conclusion?
4. If the reaction doesn't exist, is there an alternative pathway?

Respond in JSON format:
{{
    "reaction_id": "{reaction_id}",
    "strain": "{strain_name}",
    "exists": "Yes/No/Unknown/Likely",
    "confidence": "High/Medium/Low",
    "evidence": "explanation",
    "alternative": "alternative pathway or null",
    "references": ["ref1", "ref2"]
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            self._save_to_cache(cache_key, result)

            exists = result.get("exists", "Unknown")
            confidence = result.get("confidence", "Low")

            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Reaction {reaction_id} in {strain_name}: {exists} (confidence: {confidence})",
                message_ko=f"{strain_name}의 {reaction_id}: {exists} (신뢰도: {confidence})",
                data=result,
            )
        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE, message=f"LLM API error: {str(e)}", message_ko=f"LLM API 오류: {str(e)}"
            )

    def _check_gene_in_strain(self, strain_name: str = "", gene_id: str = "", **kwargs) -> SkillResult:
        """Check if a specific gene exists in a strain."""

        if not strain_name or not gene_id:
            return SkillResult(
                status=SkillStatus.FAILURE,
                message="Strain name and gene ID are required",
                message_ko="균주 이름과 유전자 ID가 필요합니다",
            )

        # Get gene name from model if available
        gene_name = gene_id
        model = self.context.appdata.project.cobra_py_model
        if model and gene_id in model.genes:
            gene = model.genes.get_by_id(gene_id)
            gene_name = gene.name or gene_id

        # Check cache
        cache_key = self._get_cache_key("check_gene", strain_name, gene_id)
        cached = self._get_cached_result(cache_key)
        if cached:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Gene {gene_id} in {strain_name} (cached)",
                message_ko=f"{strain_name}의 {gene_id} 유전자 (캐시됨)",
                data=cached,
            )

        prompt = f"""Analyze whether the following gene exists or has an ortholog in {strain_name}.

Gene ID: {gene_id}
Gene Name: {gene_name}

Please provide:
1. Does this gene or its ortholog exist in {strain_name}? (Yes/No/Unknown/Likely)
2. How confident is this assessment? (High/Medium/Low)
3. What evidence supports this conclusion?
4. If there's an ortholog, what is its ID/name in {strain_name}?
5. Brief description of the gene's function.

Respond in JSON format:
{{
    "gene_id": "{gene_id}",
    "strain": "{strain_name}",
    "exists": "Yes/No/Unknown/Likely",
    "confidence": "High/Medium/Low",
    "evidence": "explanation",
    "ortholog": "ortholog gene ID or null",
    "function": "gene function description",
    "references": ["ref1", "ref2"]
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            self._save_to_cache(cache_key, result)

            exists = result.get("exists", "Unknown")
            confidence = result.get("confidence", "Low")
            ortholog = result.get("ortholog")

            if ortholog:
                msg = f"Gene {gene_id} in {strain_name}: {exists} (ortholog: {ortholog}, confidence: {confidence})"
                msg_ko = f"{strain_name}의 {gene_id}: {exists} (오솔로그: {ortholog}, 신뢰도: {confidence})"
            else:
                msg = f"Gene {gene_id} in {strain_name}: {exists} (confidence: {confidence})"
                msg_ko = f"{strain_name}의 {gene_id}: {exists} (신뢰도: {confidence})"

            return SkillResult(status=SkillStatus.SUCCESS, message=msg, message_ko=msg_ko, data=result)
        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE, message=f"LLM API error: {str(e)}", message_ko=f"LLM API 오류: {str(e)}"
            )
