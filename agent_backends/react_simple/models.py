"""
Multi-provider model configurations for the orchestration API

Supports models from different providers via OpenRouter.
All models will be tested for tool calling capability at startup.
"""

import os
import asyncio
import logging
import os
import yaml
from typing import Dict

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Runtime validation results (populated at startup)
VALIDATED_MODELS = {}
STARTUP_VALIDATION_COMPLETE = False
# Load SUPPORTED_MODELS from config.yaml
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

SUPPORTED_MODELS = CONFIG["supported_models"]


def get_api_key(api_key_env: str) -> str:
    """Get API key from environment variable."""
    key = os.getenv(api_key_env)
    if key and key not in ["dummy-key", "your-api-key-here"]:
        logger.info(f"✅ Using API key from environment variable: {api_key_env}")
        return key
    logger.error(f"❌ Environment variable {api_key_env} not found or invalid")
    logger.info(f"💡 Set {api_key_env} in your docker-compose.yaml environment section.")
    return None

def create_llm_for_model(model_id: str) -> ChatOpenAI:
    """Create LLM instance for specific model with provider-specific configuration"""
    
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"Model {model_id} not in supported models")
    
    config = SUPPORTED_MODELS[model_id]
    api_key_env = config.get("api_key_env", "OPENROUTER_API_KEY")
    api_key = get_api_key(api_key_env)
    if not api_key:
        raise ValueError(f"No API key available for {model_id}")
    return ChatOpenAI(
        api_key=api_key,
        base_url=config["base_url"],
        model=model_id,
        temperature=0,
        streaming=True,
    )


def check_model_prerequisites(model_id: str) -> bool:
    """Check if all prerequisites for a model are met (API keys, etc.)"""
    
    if model_id not in SUPPORTED_MODELS:
        return False
    config = SUPPORTED_MODELS[model_id]
    api_key_env = config.get("api_key_env", "OPENROUTER_API_KEY")
    api_key = get_api_key(api_key_env)
    if not api_key:
        logger.warning(f"❌ Missing API key for {model_id}: Environment variable {api_key_env} not found or invalid")
        return False
    return True

async def validate_model_exists(model_id: str) -> bool:
    """Quick test to verify model exists on OpenRouter before doing full validation"""
    
    try:
        logger.info(f"🔍 Quick validation for {model_id}...")
        
        # Create LLM instance 
        test_llm = create_llm_for_model(model_id)
        
        # Make a minimal request to check if model exists
        from langchain_core.messages import HumanMessage
        
        try:
            # Simple sync generation call to test model availability
            response = await test_llm.agenerate([[HumanMessage(content="Hi")]])
            logger.info(f"✅ Model {model_id} exists and responds")
            return True
            
        except Exception as e:
            error_msg = str(e)
            provider = SUPPORTED_MODELS[model_id]["provider"]
            
            if "404" in error_msg or "No endpoints found" in error_msg:
                logger.error(f"❌ MODEL NOT FOUND: {model_id} doesn't exist on OpenRouter")
                logger.error(f"💡 Check OpenRouter documentation for correct model name")
            elif "401" in error_msg:
                logger.error(f"❌ AUTH ERROR: {model_id} - Invalid API key")
            elif "403" in error_msg:
                logger.error(f"❌ ACCESS DENIED: {model_id} - Model access not allowed")
            else:
                logger.error(f"❌ ERROR: {model_id} - {error_msg}")
            
            return False
            
    except Exception as e:
        logger.error(f"❌ CRITICAL: {model_id} - {e}")
        return False

async def test_single_model(model_id: str) -> bool:
    """Test tool calling capability for a single model with enhanced error detection"""
    
    # Check prerequisites first
    if not check_model_prerequisites(model_id):
        logger.warning(f"⏭️ Skipping {model_id} - prerequisites not met")
        return False
    
    # First, quickly validate model exists 
    if not await validate_model_exists(model_id):
        return False
    
    try:
        from response_handlers import test_tool_calling_capability
        
        logger.info(f"🧪 Testing tool calling for {model_id}...")
        start_time = asyncio.get_event_loop().time()
        
        # Create LLM instance - we know it works from validate_model_exists
        test_llm = create_llm_for_model(model_id)
        
        # Test tool calling capability
        result = await test_tool_calling_capability(test_llm, model_id)
        
        duration = (asyncio.get_event_loop().time() - start_time) * 1000
        status = "✅ PASS" if result else "❌ FAIL"
        provider = SUPPORTED_MODELS[model_id]["provider"]
        logger.info(f"{status} {model_id} ({provider}, {duration:.0f}ms)")
        
        return result
        
    except Exception as e:
        provider = SUPPORTED_MODELS[model_id]["provider"]
        logger.error(f"❌ TOOL CALLING ERROR {model_id} ({provider}): {e}")
        return False

async def validate_all_models() -> Dict[str, bool]:
    """Test tool calling capability for all supported models with prerequisites"""
    global VALIDATED_MODELS, STARTUP_VALIDATION_COMPLETE
    
    logger.info("🚀 Starting model validation...")
    
    # Check prerequisites for all models first
    testable_models = []
    skipped_models = []
    
    for model_id in SUPPORTED_MODELS.keys():
        if check_model_prerequisites(model_id):
            testable_models.append(model_id)
        else:
            skipped_models.append(model_id)
    
    logger.info(f"📋 Testing {len(testable_models)} models for tool calling capability")
    logger.info("💡 This system requires tool calling for MCP integration")
    
    if skipped_models:
        logger.info(f"⏭️ Skipping {len(skipped_models)} models (missing prerequisites): {skipped_models}")
    
    # Test models concurrently for speed
    validation_tasks = []
    for model_id in testable_models:
        task = test_single_model(model_id)
        validation_tasks.append((model_id, task))
    
    # Wait for all tests to complete
    results = {}
    for model_id, task in validation_tasks:
        try:
            result = await task
            results[model_id] = result
        except Exception as e:
            logger.error(f"❌ Validation failed for {model_id}: {e}")
            results[model_id] = False
    
    # Add skipped models as failed
    for model_id in skipped_models:
        results[model_id] = False
    
    VALIDATED_MODELS = results
    STARTUP_VALIDATION_COMPLETE = True
    
    # Enhanced summary report with providers
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    logger.info("=" * 70)
    logger.info("📊 MODEL VALIDATION SUMMARY")
    logger.info("=" * 70)
    
    # Group by provider for cleaner output
    by_provider = {}
    for model_id, result in results.items():
        provider = SUPPORTED_MODELS[model_id]["provider"]
        if provider not in by_provider:
            by_provider[provider] = []
        by_provider[provider].append((model_id, result))
    
    for provider, models in by_provider.items():
        logger.info(f"\n🏢 {provider}:")
        for model_id, result in models:
            status = "✅ READY" if result else "❌ FAILED"
            priority = SUPPORTED_MODELS[model_id]["priority"].upper()
            logger.info(f"  {status} {model_id} ({priority} priority)")
    
    logger.info("=" * 70)
    logger.info(f"🎯 OVERALL RESULT: {passed}/{total} models support tool calling")
    
    # STRICT MODE: Check if ANY models failed
    failed_models = []
    for model_id, result in results.items():
        if not result:
            failed_models.append(model_id)
    
    if failed_models:
        logger.error(f"❌ STRICT MODE: {len(failed_models)} models failed validation")
        logger.error(f"💥 Failed models: {failed_models}")
        logger.error("🛑 CONTAINER WILL EXIT - All configured models must work (strict mode)")
        logger.info("💡 Either fix the failed models or remove them from SUPPORTED_MODELS")
        
        # Import sys for exit
        import sys
        sys.exit(1)
    else:
        logger.info("✅ ALL models support tool calling - strict validation passed!")
    
    # This should never be reached now, but keeping as safety net
    if passed == 0:
        logger.error("❌ NO MODELS passed tool calling validation - system cannot function")
        logger.error("🛑 CONTAINER WILL EXIT - No working models available")
        import sys
        sys.exit(1)
    
    return results

def get_available_models():
    """Get list of models that passed validation"""
    if not STARTUP_VALIDATION_COMPLETE:
        logger.warning("⚠️ Model validation not complete - returning configured models with prerequisites")
        return [model_id for model_id in SUPPORTED_MODELS.keys() if check_model_prerequisites(model_id)]
    
    return [model for model, passed in VALIDATED_MODELS.items() if passed]

def is_model_available(model_id: str) -> bool:
    """Check if a model is available (validated and working)"""
    if not STARTUP_VALIDATION_COMPLETE:
        return model_id in SUPPORTED_MODELS and check_model_prerequisites(model_id)
    
    return VALIDATED_MODELS.get(model_id, False)

# def validate_system_configuration():
#     """Validate system configuration and test all models"""
#     # default_model = os.getenv("LLM_MODEL", "anthropic/claude-3.5-sonnet")
#     default_model = next(iter(SUPPORTED_MODELS)) 
#     logger.info("🔍 Validating system configuration...")
#     logger.info(f"📋 Default model: {default_model}")
        
#         error_msg = f"""
# ❌ CONFIGURATION ERROR: Default model '{default_model}' not in supported models

# Supported models:
# {chr(10).join(supported_list)}

# To fix this:
# 1. Update your LLM_MODEL environment variable to a supported model
# 2. Or add '{default_model}' to SUPPORTED_MODELS in config.yaml

# Current configuration:
#   LLM_MODEL={default_model}
#         """
#         logger.error(error_msg)
#         raise ValueError(f"Default model {default_model} not in supported models")
    
#     logger.info(f"✅ Default model {default_model} is in supported list")
#     return True
