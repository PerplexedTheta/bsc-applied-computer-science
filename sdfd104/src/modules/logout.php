<!-- Modules: logout.php - Jake Deery -->
<?php
session_start();
if(session_destroy()) { // End the session
    header("Location: ../"); // Redirecting To Home Page
}
?>
