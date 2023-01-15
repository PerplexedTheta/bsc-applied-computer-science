<!-- characters.php - Jake Deery -->
<?php
include('modules/session.php');

// get the id
$characterID = $_GET['id'];

// do the queries
$character_query = mysqli_query($con, "SELECT * FROM `characters` WHERE `characterID` = '" . $characterID . "'");
$character_row = mysqli_fetch_row($character_query);

$corporation_query = mysqli_query($con, "SELECT * FROM `corporation` WHERE `corporationID` = '" . $character_row[3] . "'");
$corporation_row = mysqli_fetch_row($corporation_query);

$ancestry_query = mysqli_query($con, "SELECT * FROM `ancestry` WHERE `ancestryID` = '" . $character_row[2] . "'");
$ancestry_row = mysqli_fetch_row($ancestry_query);

$faction_query = mysqli_query($con, "SELECT * FROM `faction` WHERE `factionID` = '" . $ancestry_row[1] . "'");
$faction_row = mysqli_fetch_row($faction_query);
?>
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8" />
        <title>SDFD104 BY JAKE DEERY | CHARACTER SHEET</title>
    </head>
    <body>
        <h1>SDFD104 BY JAKE DEERY | CHARACTER SHEET</h1>
        <p><b>Account name: </b><?php echo $account_row[1]; ?></p>
        <p><b>Character name: </b><?php echo $character_row[4]; ?></p>
        <p><b>Bio: </b><?php echo $character_row[5]; ?></p>
        <p><b>Gender: </b><?php
            if($character_row[7] == 0) {
            echo ' m'; } else {
            echo ' F';}
            ?></p>
        <p><b>Wallet: </b><?php echo $character_row[6]; ?></p>
        <p><b>Cosmetic Hash: </b><?php echo $character_row[8]; ?></p>
        <hr />
        <p><b>Member of: </b><?php if($character_row[3] == NULL) {
    echo("Unemployed");
} else {
    printf("<a href='corp.php?id=" . $corporation_row[0] . "'>" . $corporation_row[3] . "</a>");
} ?></p>
        <hr />
        <p><b>Faction name: </b><?php echo $faction_row[1]; ?></p>
        <p><b>Faction description: </b><?php echo $faction_row[2]; ?></p>
        <p><b>Faction iconID: </b><?php echo $faction_row[3]; ?></p>
        <hr />
        <p><b>Ancestry name: </b><?php echo $ancestry_row[2]; ?></p>
        <p><b>Ancestry description: </b><?php echo $ancestry_row[3]; ?></p>
        <p><b>Gender-specific information: </b><?php
            if($character_row[7] == 0) {
            echo $ancestry_row[4]; } else {
            echo $ancestry_row[5];}
            ?></p>
        <p><b>Ancestry attribute hex: </b><?php echo $ancestry_row[6]; ?></p>
        <p><b>Ancestry iconID: </b><?php echo $ancestry_row[7]; ?></p>
        <hr />
        <p><b>Options: </b><a href="javascript:history.go(-1);">Go back</a> - <a href="profile.php">Return to profile</a> - <a href="modules/logout.php">Logout</a></p>
    </body>
</html>
