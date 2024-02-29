mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your_heroku@email_id.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[theme]\n\
backgroundColor=\"#FFFFFF\"\n\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
