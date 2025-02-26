async function sendMessage() {
    let message = document.getElementById("userInput").value;
    
    let response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message })
    });

    let data = await response.json();
    
    let chatbox = document.getElementById("chatbox");
    chatbox.innerHTML += `<p><b>You:</b> ${message}</p>`;
    chatbox.innerHTML += `<p><b>Bot:</b> ${data.response}</p>`;
    
    document.getElementById("userInput").value = ""; // Clear input
}
