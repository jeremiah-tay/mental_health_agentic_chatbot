import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from supabase import create_client, Client
from typing import List, Dict, Tuple
from enum import Enum
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field
from dotenv import load_dotenv

# --- Environment Setup ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# --- Initialize Clients ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

print("Supabase and Groq clients initialized")

# --- CBT Technique Setup ---
class CBTTechnique(Enum):
    COGNITIVE_RESTRUCTURING = "cognitive_restructuring"
    BEHAVIORAL_ACTIVATION = "behavioral_activation"
    GROUNDING = "grounding"
    PROBLEM_SOLVING = "problem_solving"
    MINDFULNESS = "mindfulness"
    EMOTION_REGULATION = "emotion_regulation"

def fetch_technique_profiles():
    """fetch all cbt technique profiles from supabase database"""
    try:
        response = supabase.table('cbt_techniques').select('*').execute()
        
        if not response.data:
            raise Exception("No technique profiles found in database")
        
        profiles = {}
        
        for row in response.data:
            technique_key = CBTTechnique(row['technique_name'])
            
            profiles[technique_key] = {
                'description': row['description'],
                'example_phrases': row['example_phrases'],
                'indicators': row['indicators'],
                'emotional_states': row['emotional_states'],
                'when_to_use': row['when_to_use'],
                'when_not_to_use': row['when_not_to_use']
            }
        
        return profiles
        
    except Exception as e:
        print(f"Error fetching technique profiles: {str(e)}")
        raise

TECHNIQUE_PROFILES = fetch_technique_profiles()

TECHNIQUE_DISTINCTIONS = {
    "COGNITIVE_RESTRUCTURING vs MINDFULNESS": 
        "Cognitive restructuring CHALLENGES and CHANGES thoughts; mindfulness OBSERVES and ACCEPTS thoughts without changing them",
    
    "GROUNDING vs MINDFULNESS":
        "Grounding is for ACUTE distress/panic (immediate relief); mindfulness is for CHRONIC rumination/worry (long-term practice)",
    
    "BEHAVIORAL_ACTIVATION vs PROBLEM_SOLVING":
        "Behavioral activation addresses MOOD through activity; problem-solving addresses PRACTICAL issues through planning",
    
    "EMOTION_REGULATION vs GROUNDING":
        "Emotion regulation teaches SKILLS for managing intense emotions; grounding provides IMMEDIATE relief from acute distress"
}

SAFE_FALLBACK_TECHNIQUES = [
    CBTTechnique.MINDFULNESS,
    CBTTechnique.GROUNDING
]

print(f"Loaded {len(TECHNIQUE_PROFILES)} CBT technique profiles from Supabase")

# --- Load Embedding Model ---
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

technique_embeddings = {}
for technique, profile in TECHNIQUE_PROFILES.items():
    texts_to_embed = [profile["description"]] + profile["example_phrases"]
    embeddings = embedding_model.encode(texts_to_embed)
    technique_embeddings[technique] = np.mean(embeddings, axis=0)

print(f"Embedding model loaded and {len(technique_embeddings)} technique embeddings computed")

# --- Embedding-based Technique Selection ---
def select_technique_by_embedding(user_input: str, top_k: int = 3) -> List[Tuple[CBTTechnique, float]]:
    """
    select cbt techniques using semantic similarity
    returns top_k techniques with confidence scores
    """
    user_embedding = embedding_model.encode([user_input])[0]
    
    similarities = {}
    for technique, technique_embedding in technique_embeddings.items():
        similarity = cosine_similarity(
            user_embedding.reshape(1, -1),
            technique_embedding.reshape(1, -1)
        )[0][0]
        similarities[technique] = similarity
    
    sorted_techniques = sorted(
        similarities.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_k]
    
    return sorted_techniques

# --- LLM-based Technique Validation ---
def validate_technique_with_llm(user_input: str, technique: CBTTechnique) -> Tuple[bool, str]:
    """
    use llm to validate if selected technique is appropriate
    returns (is_appropriate, reasoning)
    """
    profile = TECHNIQUE_PROFILES[technique]
    
    relevant_distinctions = []
    for comparison, distinction in TECHNIQUE_DISTINCTIONS.items():
        if technique.value.upper() in comparison.upper():
            relevant_distinctions.append(f"- {comparison}: {distinction}")
    
    distinctions_text = "\n".join(relevant_distinctions) if relevant_distinctions else ""
    
    validation_prompt = f"""You are a licensed clinical psychologist specializing in Cognitive Behavioral Therapy (CBT). Your task is to validate whether a candidate CBT technique is the most appropriate and safe intervention for a specific user concern.

=== USER CONCERN ===
"{user_input}"

=== CANDIDATE TECHNIQUE ===
{technique.value.replace('_', ' ').title()}

=== TECHNIQUE PROFILE ===
description: {profile['description']}

clinical indicators for this technique:
{chr(10).join(f'  • {indicator}' for indicator in profile['indicators'])}

target emotional states:
{chr(10).join(f'  • {state}' for state in profile['emotional_states'])}

appropriate use cases:
{profile['when_to_use']}

contraindications (when NOT to use):
{profile['when_not_to_use']}

{f'=== CRITICAL DISTINCTIONS ==={chr(10)}{distinctions_text}' if distinctions_text else ''}

=== VALIDATION CRITERIA ===
evaluate the technique against these criteria in order of priority:

1. SAFETY: does the user's concern indicate any crisis elements (suicidal ideation, self-harm, psychosis, severe dissociation, medical emergency)?
   - if YES → immediately reject (appropriate: NO)
   - crisis situations require professional intervention, not automated CBT

2. CONTRAINDICATION CHECK: do any "when NOT to use" conditions clearly apply to this user's situation?
   - if YES → reject and explain why

3. CLINICAL APPROPRIATENESS: does the user's concern match the target emotional states and clinical indicators?
   - examine the user's language for markers of the emotional/cognitive/behavioral patterns this technique addresses
   - consider the intensity and acuity of symptoms

4. TECHNIQUE FIT: is this the MOST appropriate technique or would another be better?
   - use the distinctions above to differentiate between similar techniques
   - consider whether the user needs immediate relief (grounding), thought restructuring (cognitive restructuring), or behavioral change (behavioral activation)

5. THERAPEUTIC ALLIANCE: would applying this technique help build rapport or potentially harm the therapeutic relationship?
   - techniques that feel invalidating or dismissive should be rejected
   - user must be in a state to engage with the technique

=== DECISION FRAMEWORK ===
answer YES only if ALL of these are true:
- no safety concerns present
- no contraindications apply
- clinical indicators clearly match user's concern
- this is the most appropriate technique (not just "acceptable")
- user is in a state to engage with this approach

answer NO if ANY of these are true:
- safety concerns present (crisis indicators)
- contraindications clearly apply
- poor match with clinical indicators or emotional states
- another technique would be significantly more appropriate
- user's state suggests they cannot engage with this technique effectively

=== RESPONSE FORMAT ===
respond in this exact format:
APPROPRIATE: [YES/NO]
REASONING: [one clear, specific sentence citing the primary reason for your decision]

=== EXAMPLES ===

example 1:
user: "i feel like i'm a complete failure at everything i do"
technique: cognitive restructuring
APPROPRIATE: YES
REASONING: the user is expressing a global negative self-belief that is the primary target of cognitive restructuring techniques.

example 2:
user: "i can't breathe, my heart is pounding, i feel like i'm dying"
technique: cognitive restructuring
APPROPRIATE: NO
REASONING: the user is experiencing acute panic symptoms requiring immediate grounding and calming, not cognitive analysis.

example 3:
user: "i've been feeling really down and have no motivation to do anything"
technique: behavioral activation
APPROPRIATE: YES
REASONING: low mood combined with lack of motivation and inactivity are core indicators for behavioral activation interventions.

example 4:
user: "i keep worrying about everything and my mind won't stop racing"
technique: grounding
APPROPRIATE: NO
REASONING: chronic rumination and racing thoughts respond better to mindfulness practices than acute distress grounding techniques.

example 5:
user: "i don't know how to handle all these deadlines and responsibilities"
technique: problem solving
APPROPRIATE: YES
REASONING: the user is expressing feeling overwhelmed by practical challenges, which is the primary target of problem-solving therapy.

now evaluate the user concern above."""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": validation_prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        response_text = response.choices[0].message.content.strip()
        
        lines = response_text.split('\n')
        is_appropriate = False
        reasoning = "no reasoning provided"
        
        for line in lines:
            if line.startswith('APPROPRIATE:'):
                is_appropriate = 'YES' in line.upper()
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
        
        return is_appropriate, reasoning
        
    except Exception as e:
        return False, f"validation error: {str(e)}"

# --- Fallback Selection Helpers ---
def choose_best_fallback_forced(user_input: str, fallback_techniques: List[CBTTechnique]) -> CBTTechnique:
    """
    use llm to choose the most suitable fallback when all options were rejected
    compares mindfulness vs grounding to pick better fit
    """
    options_text = "\n".join([
        f"- {tech.value.replace('_', ' ').title()}: {TECHNIQUE_PROFILES[tech]['description']}"
        for tech in fallback_techniques
    ])
    
    comparison_prompt = f"""you are a licensed clinical psychologist specializing in Cognitive Behavioral Therapy (CBT). All primary CBT techniques have been rejected for this user's concern, and you must now select the safest and most appropriate fallback technique.

=== SITUATION ===
user concern: "{user_input}"

=== AVAILABLE FALLBACK OPTIONS ===
{options_text}

=== SELECTION CRITERIA ===

these are SAFE FALLBACK techniques - universally appropriate and low-risk interventions that can provide support when more specific techniques are contraindicated.

Mindfulness:
Best for:
- chronic worry and rumination that is not acute
- racing thoughts without panic symptoms
- general emotional dysregulation
- situations where the user needs to observe thoughts/feelings without judgment
- longer-term practice for anxiety management

when to choose:
- user describes ongoing mental patterns (constant worrying, can't stop thinking)
- user needs acceptance-based approach
- situation is not an acute crisis
- user can engage in reflective practice

Grounding:
best for:
- acute panic attacks or anxiety spikes
- dissociation or feeling disconnected from reality
- overwhelming emotions requiring immediate relief
- feeling out of control in the present moment
- need for rapid symptom reduction

when to choose:
- user describes intense physical symptoms (heart racing, can't breathe, dizziness)
- user needs immediate calming and stabilization
- situation has acute/crisis quality
- user needs to reconnect with present moment quickly

=== KEY DISTINCTION ===
Mindfulness = CHRONIC patterns (ongoing worry/rumination) → observation and acceptance
Grounding = ACUTE distress (panic/overwhelm) → immediate relief and stabilization

if unsure, consider:
- does the user use words like "always", "constantly", "keep" → mindfulness
- does the user use words like "right now", "can't breathe", "losing control" → grounding
- is this happening over time or in this moment? over time = mindfulness, moment = grounding

=== DECISION FRAMEWORK ===
1. identify the temporal quality: is this acute (happening now) or chronic (ongoing pattern)?
2. identify the intensity: is this overwhelming and destabilizing or manageable but persistent?
3. identify the need: does user need immediate relief or long-term coping?

acute + high intensity + immediate relief needed → grounding
chronic + moderate intensity + long-term coping needed → mindfulness

=== RESPONSE FORMAT ===
respond in this exact format:
SELECTED: [technique name exactly as shown above - either "Mindfulness" or "Grounding"]
REASON: [one clear sentence explaining why this technique better matches the user's temporal pattern and intensity level]

=== EXAMPLES ===

example 1:
user: "i keep worrying about everything and my mind won't stop racing"
SELECTED: Mindfulness
REASON: the user describes a chronic pattern of worry and racing thoughts that requires observation and acceptance rather than acute intervention.

example 2:
user: "i can't breathe, my heart is pounding, i feel like i'm going to die right now"
SELECTED: Grounding
REASON: the user is experiencing acute panic symptoms requiring immediate stabilization and present-moment reconnection.

example 3:
user: "i always feel anxious and overthink every decision"
SELECTED: Mindfulness
REASON: the ongoing pattern of anxiety and overthinking responds better to mindfulness-based observation than acute grounding techniques.

example 4:
user: "everything feels unreal and i feel like i'm floating outside my body"
SELECTED: Grounding
REASON: the dissociative symptoms require immediate grounding techniques to reconnect with physical reality and the present moment.

example 5:
user: "i'm stressed about my life and feel overwhelmed by everything going on"
SELECTED: Mindfulness
REASON: the general overwhelm and stress suggest chronic pressure rather than acute crisis, making mindfulness more appropriate for building coping capacity.

now evaluate the user concern above and select the most appropriate fallback technique."""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": comparison_prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        response_text = response.choices[0].message.content.strip()
        
        for line in response_text.split('\n'):
            if line.startswith('SELECTED:'):
                selected_name = line.replace('SELECTED:', '').strip().lower()
                for tech in fallback_techniques:
                    if tech.value.replace('_', ' ') in selected_name or tech.value in selected_name:
                        return tech
        
        return fallback_techniques[0]
    
    except Exception as e:
        return fallback_techniques[0]

def choose_best_fallback_approved(user_input: str, approved_fallbacks: List[Dict]) -> CBTTechnique:
    """
    use llm to choose best technique among multiple approved fallbacks
    """
    options_text = "\n".join([
        f"- {fb['technique'].replace('_', ' ').title()}: {fb['reasoning']}"
        for fb in approved_fallbacks
    ])
    
    selection_prompt = f"""You are a licensed clinical psychologist specializing in Cognitive Behavioral Therapy (CBT). multiple safe fallback techniques have been validated as appropriate for this user's concern. you must now select the MOST appropriate option.

=== SITUATION ===
user concern: "{user_input}"

=== APPROVED FALLBACK OPTIONS ===
{options_text}

=== SELECTION TASK ===
both techniques are clinically appropriate (they passed validation), but you must select the one that is the BEST FIT for this specific situation.

consider these factors in priority order:

1. TEMPORAL MATCH (most important):
   - is the user's concern acute (happening now) or chronic (ongoing pattern)?
   - acute symptoms → prefer grounding for immediate relief
   - chronic patterns → prefer mindfulness for long-term skill building

2. SYMPTOM INTENSITY:
   - high intensity, destabilizing symptoms → prefer grounding
   - moderate intensity, manageable symptoms → prefer mindfulness

3. USER'S STATED NEED:
   - if user implies need for immediate relief → prefer grounding
   - if user implies need for coping strategies → prefer mindfulness

4. LANGUAGE MARKERS:
   acute markers: "right now", "currently", "can't", "losing control", "overwhelming" → grounding
   chronic markers: "always", "keep", "constantly", "every time", "never stops" → mindfulness

=== DECISION FRAMEWORK ===
ask yourself:
- which technique better matches the temporal quality of the user's concern?
- which technique addresses the intensity level described?
- which technique meets the implied need (immediate vs long-term)?

select the technique that has the strongest match across these dimensions.

=== RESPONSE FORMAT ===
respond in this exact format:
SELECTED: [technique name exactly as shown in the options above]
REASON: [one clear sentence citing the primary factor (temporal, intensity, or need) that makes this technique the better fit]

=== EXAMPLES ===

example 1:
user: "i keep overthinking and worrying about everything all the time"
approved options:
- Mindfulness: appropriate for chronic worry patterns requiring acceptance
- Grounding: appropriate for managing anxiety symptoms
SELECTED: Mindfulness
REASON: the chronic pattern indicated by "keep" and "all the time" better matches mindfulness for long-term worry management.

example 2:
user: "my heart is racing and i feel like i'm going to pass out"
approved options:
- Mindfulness: appropriate for anxiety symptom management
- Grounding: appropriate for acute panic symptoms requiring stabilization
SELECTED: Grounding
REASON: the acute physical symptoms and immediate distress require grounding's rapid stabilization over mindfulness practice.

now evaluate the user concern above and select the most appropriate technique from the approved options."""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": selection_prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        response_text = response.choices[0].message.content.strip()
        
        for line in response_text.split('\n'):
            if line.startswith('SELECTED:'):
                selected_name = line.replace('SELECTED:', '').strip().lower()
                for fb in approved_fallbacks:
                    if fb['technique'].replace('_', ' ') in selected_name or fb['technique'] in selected_name:
                        return CBTTechnique(fb['technique'])
        
        return CBTTechnique(approved_fallbacks[0]['technique'])
    
    except Exception as e:
        return CBTTechnique(approved_fallbacks[0]['technique'])

# --- Pydantic Model for Tool Input ---
class SelectCBTTechniqueArgs(BaseModel):
    user_mental_health_concern: str = Field(..., description="The user's mental health concern that needs CBT technique selection")

# --- CBT Technique Selection Tool ---
@tool("select_cbt_technique", args_schema=SelectCBTTechniqueArgs)
def select_cbt_technique(user_mental_health_concern: str) -> dict:
    """
    selects the most appropriate cbt technique for a user's mental health concern.
    
    use this tool ONLY when ALL of these conditions are met:
    1. user is expressing a mental health concern (anxiety, depression, stress, negative thoughts, panic, worry, overwhelm, low mood, procrastination, etc.)
    2. the concern is appropriate for brief cbt intervention (NOT crisis, suicidal ideation, psychosis, eating disorders, or medical emergencies)
    3. the user message has enough detail to understand their issue (NOT just single vague words like "help", "bad", "stressed" with no context)
    
    do NOT use this tool when:
    - user is in crisis or mentions suicide, self-harm, severe symptoms requiring professional help
    - user is making casual conversation (greetings, small talk, non-mental-health questions)
    - user message is too vague or ambiguous (ask for clarification first)
    - you can provide support without needing technique selection
    
    the tool returns the selected cbt technique with validation metadata that you should use to guide your therapeutic response.
    """
    #step 1: embedding-based selection
    embedding_results = select_technique_by_embedding(user_mental_health_concern, top_k=3)
    
    validation_attempts = []
    selected_technique = None
    
    #step 2: validate top 3 techniques
    for technique, similarity_score in embedding_results:
        is_appropriate, reasoning = validate_technique_with_llm(user_mental_health_concern, technique)
        
        validation_attempts.append({
            "technique": technique.value,
            "similarity_score": float(similarity_score),
            "is_appropriate": is_appropriate,
            "reasoning": reasoning
        })
        
        if is_appropriate:
            selected_technique = technique
            break
    
    #step 3: fallback if all rejected
    if selected_technique is None:
        fallback_validations = []
        
        for fallback_technique in SAFE_FALLBACK_TECHNIQUES:
            is_appropriate, reasoning = validate_technique_with_llm(user_mental_health_concern, fallback_technique)
            
            fallback_validations.append({
                "technique": fallback_technique.value,
                "similarity_score": 0.0,
                "is_appropriate": is_appropriate,
                "reasoning": reasoning,
                "is_fallback": True
            })
        
        validation_attempts.extend(fallback_validations)
        
        approved_fallbacks = [fv for fv in fallback_validations if fv["is_appropriate"]]
        
        if len(approved_fallbacks) == 0:
            selected_technique = choose_best_fallback_forced(user_mental_health_concern, SAFE_FALLBACK_TECHNIQUES)
            validation_attempts.append({
                "technique": selected_technique.value,
                "similarity_score": 0.0,
                "is_appropriate": False,
                "reasoning": f"forced fallback - chose {selected_technique.value} as more suitable than alternatives",
                "is_fallback": True,
                "is_forced": True
            })
        elif len(approved_fallbacks) == 1:
            selected_technique = CBTTechnique(approved_fallbacks[0]["technique"])
        else:
            selected_technique = choose_best_fallback_approved(user_mental_health_concern, approved_fallbacks)
            validation_attempts.append({
                "technique": selected_technique.value,
                "similarity_score": 0.0,
                "is_appropriate": True,
                "reasoning": f"selected {selected_technique.value} as most suitable among approved fallbacks",
                "is_fallback": True,
                "is_best_choice": True
            })
    
    return {
        "selected_technique": selected_technique.value,
        "technique_description": TECHNIQUE_PROFILES[selected_technique]['description'],
        "when_to_use": TECHNIQUE_PROFILES[selected_technique]['when_to_use'],
        "embedding_results": [(tech.value, float(score)) for tech, score in embedding_results],
        "validation_attempts": validation_attempts
    }

print("CBT technique selection tool initialized and ready for import")
