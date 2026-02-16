"""
Summarization prompt templates for Italian and English.

All prompts follow the Llama-3.1 chat format with system/user/assistant headers.
Three prompt styles are available:
  - zero_shot: Direct summarization instruction
  - few_shot: With illustrative examples (IT only)
  - source_aware: Anti-hallucination prompt with strict fidelity rules
"""

SUMMARY_PROMPTS = {
    # -------------------------------------------------------------------------
    # ITALIAN PROMPTS
    # -------------------------------------------------------------------------
    "it": {
        "zero_shot": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "Sei un esperto di sintesi testuale italiano. Il tuo compito è generare riassunti\n"
            "che siano:\n"
            "- FEDELI: Non inventare informazioni non presenti nel testo\n"
            "- COMPLETI: Includi tutti i concetti chiave indicati\n"
            "- COERENTI: Il riassunto deve essere fluido e ben scritto\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "TESTO ORIGINALE:\n{source}\n\n---\n\n"
            "ISTRUZIONI OBBLIGATORIE:\n"
            "I seguenti concetti chiave sono stati estratti dal testo e DEVONO essere\n"
            "riflessi nel riassunto finale:\n\n{keyphrases}\n\n---\n\n"
            "Genera un riassunto astrattivo in italiano (150-250 parole):\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        ),

        "few_shot": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "# RUOLO\nSei un assistente editoriale esperto specializzato nella sintesi di testi complessi.\n\n"
            "# OBIETTIVO\nGenerare riassunti astrattivi in Italiano che integrino perfettamente "
            "le informazioni salienti fornite (Frasi Chiave).\n\n"
            "# VINCOLI E PRIMING\n"
            "1. **Content Priming**: Le Frasi Chiave sono i pilastri del contenuto. Devono essere tutte presenti.\n"
            "2. **Structural Priming**: Usa le Frasi Chiave come **scaletta logica** per strutturare il paragrafo.\n"
            "3. **Stile**: Mantieni un tono formale, accademico e oggettivo.\n"
            "4. **Formato**: Segui rigorosamente la struttura degli esempi forniti.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "### Esempio 1\n"
            "Testo: \"La rivista ''Huey, Dewey and Louie Junior Woodchucks'', che pubblicava "
            "le avventure delle Giovani Marmotte, per un certo periodo vide la collaborazione "
            "alternata ai testi di Jerry Siegel e Barks. Soprattutto il secondo, tornato in attività "
            "dopo essersi ufficialmente ritirato su richiesta di Chase Craig - a quel tempo curatore "
            "editoriale della Western Publishing - realizzò molte storie generalmente disegnate da "
            "altri artisti. In particolare questa venne disegnata da Kay Wright e pubblicata sul 12° "
            "numero della collana nel gennaio 1972. Ne venne poi realizzata una nuova versione dal "
            "disegnatore Daan Jippes pubblicata sul settimanale olandese ''Donald Duck'' il 27 marzo "
            "del 1992 e pubblicata in Italia come '''Le Giovani Marmotte e la danza tempestosa'''.\"\n"
            "Frasi Chiave: Giovani Marmotte, Carl Barks, Kay Wright, 1972, Daan Jippes, 1992\n"
            "Riassunto: \"''Le Giovani Marmotte e la danza 'antiscolastica''' è una storia a fumetti "
            "scritta da Carl Barks e disegnata da Kay Wright. Venne pubblicata nel 1972, e Daan Jippes "
            "ne realizzò una nuova versione nel 1992.\"\n\n"
            "### Esempio 2\n"
            "Testo: \"La ''Lega nazionale degli studenti greci in Italia'' venne fondata a Roma nel "
            "giugno del 1967, in seguito al colpo di Stato dei colonnelli greci del 21 aprile dello "
            "stesso anno. Ebbe sedi in tutte le principali città universitarie italiane. L'Esesi fu "
            "subito punto di riferimento della giunta militare golpista di Atene. Venne incaricata "
            "dell'attività propagandistica in favore del nuovo governo e della 'vigilanza morale sul "
            "credo nazionale degli studenti greci in Italia'.\"\n"
            "Frasi Chiave: Lega nazionale degli studenti greci in Italia, regime dei colonnelli greci, "
            "movimento, Italia\n"
            "Riassunto: \"La Lega nazionale degli studenti greci in Italia fu un movimento operante in "
            "Italia a favore del regime dei colonnelli greci.\"\n\n---\n\n"
            "### TASK CORRENTE\nTesto Originale:\n{source}\n\n"
            "Frasi Chiave (da usare come ossatura):\n{keyphrases}\n\n"
            "Genera il riassunto collegando logicamente i punti chiave:\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        ),

        "source_aware": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "Sei un riassuntore FEDELE di articoli enciclopedici. I tuoi riassunti devono contenere "
            "SOLO fatti presenti nella fonte.\n\n"
            "REGOLE CRITICHE DI FEDELTÀ:\n"
            "1. Usa SOLO informazioni esplicitamente presenti nel testo fornito.\n"
            "2. NON aggiungere conoscenze esterne, date, fatti o nomi non presenti nella fonte.\n"
            "3. Se la fonte non menziona qualcosa, NON devi menzionarlo nemmeno tu.\n"
            "4. In caso di dubbio su un fatto, OMETTILO piuttosto che inventarlo.\n"
            "5. NON inferire o estrapolare oltre quanto scritto.\n\n"
            "REGOLE DI SCRITTURA:\n"
            "1. Scrivi UN SOLO paragrafo (50-80 parole). NO titoli, NO elenchi.\n"
            "2. SINTETIZZA e RIFORMULA - mai copiare frasi letteralmente.\n"
            "3. Inizia con una definizione o contestualizzazione del soggetto.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "TESTO FONTE:\n{source}\n\n"
            "CONCETTI CHIAVE DA INTEGRARE (tutti dalla fonte sopra):\n{keyphrases}\n\n"
            "Scrivi un riassunto FEDELE usando SOLO fatti dalla fonte sopra. "
            "Non aggiungere conoscenze esterne.\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        ),
    },

    # -------------------------------------------------------------------------
    # ENGLISH PROMPTS
    # -------------------------------------------------------------------------
    "en": {
        "zero_shot": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are an expert text summarizer. Your task is to generate summaries that are:\n"
            "- FAITHFUL: Do not invent information not present in the text\n"
            "- COMPLETE: Include all indicated key concepts\n"
            "- COHERENT: The summary must be fluid and well-written\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "ORIGINAL TEXT:\n{source}\n\n---\n\n"
            "MANDATORY INSTRUCTIONS:\n"
            "The following key concepts were extracted from the text and MUST be\n"
            "reflected in the final summary:\n\n{keyphrases}\n\n---\n\n"
            "Generate an abstractive summary in English (150-250 words):\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        ),

        "source_aware": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a FAITHFUL summarizer of scientific papers. Your summaries must contain "
            "ONLY facts present in the source.\n\n"
            "CRITICAL FIDELITY RULES:\n"
            "1. Use ONLY information explicitly present in the provided text.\n"
            "2. Do NOT add external knowledge, dates, facts, or names not in the source.\n"
            "3. If the source doesn't mention something, do NOT mention it either.\n"
            "4. When in doubt about a fact, OMIT it rather than inventing it.\n"
            "5. Do NOT infer or extrapolate beyond what is written.\n\n"
            "WRITING RULES:\n"
            "1. Write ONE SINGLE paragraph (50-80 words). NO titles, NO lists.\n"
            "2. SYNTHESIZE and REPHRASE - never copy sentences verbatim.\n"
            "3. Begin with a definition or contextualization of the subject.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "SOURCE TEXT:\n{source}\n\n"
            "KEY CONCEPTS TO INTEGRATE (all from the source above):\n{keyphrases}\n\n"
            "Write a FAITHFUL summary using ONLY facts from the source above. "
            "Do not add external knowledge.\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        ),
    },
}


def get_summary_prompt(lang: str, prompt_type: str) -> str:
    """Get a summarization prompt template.

    Args:
        lang: Language code ("it" or "en").
        prompt_type: One of "zero_shot", "few_shot", "source_aware".

    Returns:
        Prompt template string with {source} and {keyphrases} placeholders.

    Raises:
        ValueError: If lang or prompt_type is not available.
    """
    if lang not in SUMMARY_PROMPTS:
        raise ValueError(f"No prompts for language '{lang}'. Available: {list(SUMMARY_PROMPTS.keys())}")
    lang_prompts = SUMMARY_PROMPTS[lang]
    if prompt_type not in lang_prompts:
        raise ValueError(f"No '{prompt_type}' prompt for '{lang}'. Available: {list(lang_prompts.keys())}")
    return lang_prompts[prompt_type]
