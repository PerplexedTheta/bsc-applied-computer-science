<!-- profile.php - Jake Deery -->
<?php
include('modules/session.php');
?>
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8" />
        <title>SDFD104 BY JAKE DEERY | PROFILE</title>
    </head>
    <body>
        <h1>SDFD104 BY JAKE DEERY | PROFILE</h1>
        <p><b>Account name: </b><?php echo $account_row[1]; ?></p>
        <p><b>Account type: </b><?php
            if($account_row[3] != 99999) {
            echo 'Standard Account'; } else {
            echo 'Game Moderator';}
            ?></p>
        <p><b>Online?</b><?php
            if($account_row[4] == 0) {
            echo ' N'; } else {
            echo ' y';}
            ?></p>
        <p><b>Banned?</b><?php
            if($account_row[5] == 0) {
            echo ' N'; } else {
            echo ' y';}
            ?></p>
        <hr />
        <p>List of available characters:</p>
        <?php
        if ($result = mysqli_query($con, "SELECT * FROM `characters` WHERE `accountID` = '" . $session_store . "'")) {
            while ($row = mysqli_fetch_row($result)) {
                printf("<a href='characters.php?id=" . $row[0] . "'>" . $row[4] . "</a><br />");
            }
            
            mysqli_free_result($result);
        }
        ?>
        <hr />
        <p><b>Options: </b><a href="javascript:history.go(-1);">Go back</a> - <a href="modules/logout.php">Logout</a></p>
    </body>
</html>
