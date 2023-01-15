<!-- Modules: session.php - Jake Deery -->
<?php
if (!isset($_SESSION)) {
    session_start();
}

// define connection details
$con = mysqli_connect("127.0.0.1", "username", "passwd", "dbname");

// check connection
if (mysqli_connect_errno()) {
    printf("Connect failed: %s\n", mysqli_connect_error());
    exit();
}

// store session
$session_store = $_SESSION['sdfd104'];

// do the queries
$account_query = mysqli_query($con, "SELECT * FROM `account` WHERE `accountID` = '" . $session_store . "'");
$account_row = mysqli_fetch_row($account_query);

if(!isset($session_store)){
    mysqli_close($con); // Closing Connection
    header('Location: ./'); // Redirecting To Home Page
}
?>
