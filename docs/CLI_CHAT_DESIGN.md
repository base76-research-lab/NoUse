# Goose-inspirerat CLI-chat för NoUse

## Mål
- Ett enda kommando (`nouse chat`) startar en interaktiv chat med NoUse-minne och valfri LLM (lokal eller API).
- Ingen konfiguration krävs vid start.
- Automatisk upptäckt av lokal modell (Ollama/LM Studio) eller API-nyckel (OpenAI).
- Tydlig feedback om ingen modell hittas.
- Enkel, barnvänlig terminal-REPL.

## Flöde
1. Användaren kör `nouse chat`.
2. CLI försöker:
   - Hitta lokal Ollama/LM Studio-modell.
   - Om ingen hittas: fråga efter OpenAI-nyckel.
3. Startar REPL:
   - Prompt: "Fråga vad du vill (eller skriv 'quit' för att avsluta):"
   - Svar visas direkt, med info om svaret kommer från minnet eller LLM.
4. Om ingen modell hittas eller API-nyckel anges: visa enkel instruktion.

## Teknik
- Byggs som nytt kommando i `nouse.cli.main` (t.ex. `@app.command("chat")`).
- Modellupptäckt i Python (försök ansluta till Ollama/LM Studio, annars fråga efter OpenAI-nyckel).
- Enkel REPL-loop med input()/print().
- Använd befintlig NoUse-minnesintegration.
- Tydliga felmeddelanden och exit-vägar.

## Test
- Smoke-test: Starta kommandot, skriv en fråga, verifiera att svar ges.
- Testa fallback till API-nyckel.
- Testa felhantering (ingen modell, ingen nyckel).

## Nästa steg
1. Skapa CLI-kommando och REPL.
2. Implementera modellupptäckt.
3. Integrera NoUse-minne.
4. Testa och verifiera.
5. Commit/push.
