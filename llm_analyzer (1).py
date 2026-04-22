from groq import Groq
import os, json, re

SYSTEM_PROMPT = """You are an elite technical recruiter and career strategist with 20+ years of experience at top-tier companies (FAANG, McKinsey, Goldman Sachs). You have reviewed over 50,000 resumes.

Your analysis is:
- HYPER-SPECIFIC: Reference actual details from the resume — company names, project names, exact technologies, numbers
- BRUTALLY HONEST: Do not sugarcoat gaps or weaknesses
- DEEPLY PERSONALIZED: Every word is specific to THIS candidate for THIS role — never generic
- ACTIONABLE: Every recommendation has a concrete, implementable next step

Respond ONLY with valid JSON. No markdown. No preamble. No trailing text."""

ANALYSIS_PROMPT = """Evaluate this specific candidate for this specific role.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

RAG RELEVANT SECTIONS:
{relevant_sections}

RAG Semantic Score: {semantic_score}/100
{linkedin_section}

Return ONLY this JSON structure:
{{
  "overall_match_score": <0-100>,
  "confidence_level": "<Low|Medium|High>",
  "verdict": "<Exceptional Candidate|Strong Match|Solid Candidate|Moderate Match|Weak Match|Poor Fit>",
  "executive_summary": "<4-5 sentences referencing actual companies, roles, technologies by name. Write as a recruiter private assessment note.>",
  "skills_analysis": {{
    "matched_skills": [{{"skill":"<name>","evidence":"<where in resume>","proficiency":"<Beginner|Intermediate|Advanced|Expert>"}}],
    "missing_skills": [{{"skill":"<name>","importance":"<Critical|Important|Nice-to-have>","learnability":"<Easy|Moderate|Hard>"}}],
    "bonus_skills": [{{"skill":"<name>","value":"<why this adds unique value>"}}],
    "skill_coverage_percent": <0-100>
  }},
  "experience_analysis": {{
    "years_required": "<from JD or Not specified>",
    "years_detected": "<computed from resume dates>",
    "experience_match": "<Significantly Exceeds|Exceeds|Meets|Partially Meets|Does Not Meet>",
    "career_trajectory": "<Ascending|Lateral|Descending|Mixed>",
    "relevant_roles": [{{"title":"<role>","company":"<company>","relevance":"<why this matters for the JD>"}}],
    "industry_alignment": "<detailed assessment>",
    "biggest_experience_gap": "<the single most important missing experience>"
  }},
  "education_analysis": {{
    "required": "<from JD or Not specified>",
    "candidate": "<exact degree and institution from resume>",
    "education_match": "<Exceeds|Meets|Partially Meets|Does Not Meet>",
    "notable_academic_signals": "<notable schools, GPA, honors, relevant coursework>"
  }},
  "strengths": [{{"title":"<3-5 word title>","detail":"<2-3 sentences with specific resume evidence>"}}],
  "gaps": [{{"title":"<3-5 word title>","severity":"<Critical|Significant|Minor>","detail":"<specific gap with context>"}}],
  "hidden_strengths": ["<non-obvious advantage the candidate has that they may have undersold>"],
  "red_flags": [{{"flag":"<concern>","severity":"<Minor|Moderate|Serious>","context":"<fair interpretation>"}}],
  "personality_fit": {{
    "inferred_work_style": "<what the resume signals about how they work>",
    "culture_signals": "<what companies and projects suggest about cultural fit>",
    "communication_level": "<signal of articulation from how they present themselves>"
  }},
  "recommendations": [
    {{
      "priority": "<High|Medium|Low>",
      "category": "<Resume|Skills|Experience|Framing|Networking|Interview Prep>",
      "action": "<specific implementable recommendation — not generic advice>",
      "timeline": "<Immediate|1-2 weeks|1-3 months|3-6 months>"
    }}
  ],
  "interview_prep": {{
    "likely_questions": [{{"question":"<specific question the interviewer will ask this candidate>","why":"<why based on resume and JD>"}}],
    "talking_points_to_emphasize": ["<specific thing from resume to highlight>"],
    "topics_to_avoid_or_prepare_for": ["<potential weak spot to have answers ready for>"]
  }},
  "keyword_density": {{
    "high_impact_keywords_found": ["<keyword>"],
    "missing_keywords": ["<keyword that should appear in resume for ATS>"],
    "ats_optimization_tips": "<specific advice for improving ATS pass rate for this exact JD>"
  }},
  "compensation": {{
    "estimated_current_band": "<salary range estimate in INR based on role and experience>",
    "negotiation_leverage": "<High|Medium|Low>",
    "leverage_reason": "<why they have or lack leverage>"
  }},
  "ats_score": <0-100>,
  "interview_probability": "<Very Low|Low|Moderate|High|Very High>",
  "hiring_recommendation": "<Reject|Maybe|Phone Screen|Technical Interview|Final Round|Immediate Offer>"
}}"""


class LLMAnalyzer:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set.\n"
                "Run: $env:GROQ_API_KEY='gsk_your_key_here'"
            )
        self.client = Groq(api_key=api_key)

    def analyze(self, resume_text, job_description, relevant_sections, semantic_score, linkedin_url=None):
        sections_str = "\n\n".join(
            f"[Relevance {s['score']*100:.0f}%]\n{s['text']}"
            for s in relevant_sections
        )

        linkedin_section = ""
        if linkedin_url and linkedin_url.strip():
            linkedin_section = f"""
CANDIDATE LINKEDIN URL: {linkedin_url}
Include a "linkedin_analysis" object in your JSON with these fields:
  profile_found: true,
  profile_strength: "<Weak|Moderate|Strong|All-Star>",
  consistency_with_resume: "<does LinkedIn likely match the resume claims>",
  linkedin_specific_strengths: ["<strength>"],
  missing_from_linkedin: ["<what should be added>"],
  recommendations: ["<specific LinkedIn improvement>"]
"""

        prompt = ANALYSIS_PROMPT.format(
            job_description=job_description[:2000],
            resume_text=resume_text[:2500],
            relevant_sections=sections_str,
            semantic_score=semantic_score,
            linkedin_section=linkedin_section
        )

        try:
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=2000,
                temperature=0.3,
                top_p=0.9,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ]
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```\s*$', '', raw)
            result = json.loads(raw)
            result["semantic_score"] = semantic_score
            result["rag_sections"] = relevant_sections[:3]
            return result

        except json.JSONDecodeError as e:
            match = re.search(r'\{[\s\S]*\}', raw if 'raw' in dir() else '{}')
            if match:
                try:
                    r = json.loads(match.group())
                    r["semantic_score"] = semantic_score
                    r["rag_sections"] = relevant_sections[:3]
                    return r
                except Exception:
                    pass
            return self._fallback(semantic_score, str(e))

        except Exception as e:
            raise RuntimeError(f"Analysis failed: {e}")

    def _fallback(self, semantic_score, error):
        return {
            "overall_match_score": int(semantic_score),
            "confidence_level": "Low",
            "verdict": "Analysis Incomplete",
            "executive_summary": "Could not parse AI response. Please try again.",
            "skills_analysis": {"matched_skills": [], "missing_skills": [], "bonus_skills": [], "skill_coverage_percent": 0},
            "experience_analysis": {"years_required": "N/A", "years_detected": "N/A", "experience_match": "Unknown",
                "career_trajectory": "Unknown", "relevant_roles": [], "industry_alignment": "Unknown", "biggest_experience_gap": "Unknown"},
            "education_analysis": {"required": "N/A", "candidate": "N/A", "education_match": "Unknown", "notable_academic_signals": "N/A"},
            "strengths": [], "gaps": [], "hidden_strengths": [], "red_flags": [],
            "personality_fit": {"inferred_work_style": "Unknown", "culture_signals": "Unknown", "communication_level": "Unknown"},
            "recommendations": [],
            "interview_prep": {"likely_questions": [], "talking_points_to_emphasize": [], "topics_to_avoid_or_prepare_for": []},
            "keyword_density": {"high_impact_keywords_found": [], "missing_keywords": [], "ats_optimization_tips": "N/A"},
            "compensation": {"estimated_current_band": "Unknown", "negotiation_leverage": "Unknown", "leverage_reason": "Unknown"},
            "ats_score": int(semantic_score * 0.85),
            "interview_probability": "Unknown",
            "hiring_recommendation": "Unknown",
            "semantic_score": semantic_score,
            "rag_sections": [],
            "_parse_error": error
        }