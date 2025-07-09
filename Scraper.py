# scraper.py (Versão 7.0 - A Abordagem Pragmática)

import sys
import pyperclip
from playwright.sync_api import sync_playwright, TimeoutError
from fake_useragent import UserAgent

def scrape_and_copy(url: str):
    """
    Abandona a espera por 'networkidle' e usa um tempo de espera fixo,
    uma abordagem mais robusta para sites que nunca param de fazer requisições.
    """
    try:
        with sync_playwright() as p:
            ua = UserAgent()
            user_agent_string = ua.random
            
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=user_agent_string)
            page = context.new_page()
            
            # 1. Navega para a página, mas só espera a estrutura base (muito rápido)
            page.goto(url, timeout=90000, wait_until="domcontentloaded")

            # 2. A MUDANÇA ESSENCIAL: Espera por um tempo fixo
            # Damos 7 segundos para os scripts do site (React, etc.) renderizarem
            # o conteúdo na tela. É uma simulação de paciência humana.
            page.wait_for_timeout(7000)

            # 3. Extrai o conteúdo visível após a espera
            content = page.locator('body').inner_text()
            browser.close()

            if not content or not content.strip():
                raise ValueError("Nenhum conteúdo de texto foi encontrado após a espera.")

            pyperclip.copy(content)
            return "SUCCESS"
            
    except Exception as e:
        error_message = f"SCRAPER_ERROR: {str(e)}"
        pyperclip.copy(error_message)
        return error_message

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url_to_scrape = sys.argv[1]
        result = scrape_and_copy(url_to_scrape)
        print(result)